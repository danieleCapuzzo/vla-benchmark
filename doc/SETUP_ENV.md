# Environment Setup — VLA-on-LIBERO Thesis

### Complete rebuild guide + provenance — RTX 5080 (Blackwell sm_120), Pop!\_OS 24.04

> Single self-contained document. Covers: system prerequisites, the two Python environments
> (`.openvla-env`, `.oft-env`), shared LIBERO, folder structure, and full provenance
> (versions, commit hashes, deviations, bugs encountered).
> Last end-to-end verification: 2026-07-07 — real OFT inference on LIBERO-Spatial working,
> LIBERO installed and verified in both venvs.
>
> **Mandatory order:** System → Python → OpenVLA → OFT → LIBERO (in both venvs).

---

# PART I — FOLDER STRUCTURE

```
~/Desktop/thesis-vla/
├── .openvla-env/                 # OpenVLA base venv (Python 3.10)
├── .oft-env/                     # OpenVLA-OFT venv (Python 3.10)
├── LIBERO/                       # single shared clone, installed editable in BOTH venvs
└── vla-benchmark/                # working Git repo
    ├── openvla/                   # official openvla/openvla clone (read-only)
    ├── openvla-oft/                # official moojink/openvla-oft clone (read-only)
    ├── mail/                       # MaIL clone (read-only, separate .mail-env — not yet built)
    ├── doc/
    │   ├── ARCHITECTURE.md         # design contract
    │   ├── SETUP_ENV.md            # this file
    │   └── pip-lists/              # exported pip freezes per environment
    ├── core/ wrappers/ config/ results_io/ tests/   # the testbench package itself
    ├── run_eval.py                 # CLI entrypoint
    └── .vscode/
        └── settings.json           # interpreter pointed at the correct venv
```

**Structure notes:**

- `.openvla-env` and `.oft-env` live in `thesis-vla/` (one level above `vla-benchmark/`).
- `LIBERO/` is cloned once and installed editable (`pip install -e .`) into both venvs: one
  source of truth, one commit to track.
- `openvla/`, `openvla-oft/`, and `mail/` live INSIDE `vla-benchmark/`. They are separate Git
  repos with their own history — `vla-benchmark/.gitignore` excludes them; only their commit
  hashes are tracked here (Part VI).

**VS Code `settings.json`** (in `vla-benchmark/.vscode/`, for the "open vla-benchmark as
root" scenario):

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/../.openvla-env/bin/python",
  "python.terminal.activateEnvironment": true,
  "editor.formatOnSave": true,
  "files.exclude": { "**/__pycache__": true, "**/*.pyc": true },
  "terminal.integrated.env.linux": {
    "PATH": "${workspaceFolder}/../.openvla-env/bin:${env:PATH}"
  }
}
```

To work on OFT, change both paths to `.oft-env`. Alternatively use a multi-root workspace
with two separate `.vscode/settings.json` (one for `openvla/`, one for `openvla-oft/`).

---

# PART II — SYSTEM PREREQUISITES (one-time)

## II.1 NVIDIA driver + CUDA Toolkit 12.8

```bash
nvidia-smi   # verify driver >= 570 and the RTX 5080 is visible
```

Install CUDA Toolkit 12.8 (NOT 12.9 or 13.0 — known Blackwell bugs on those versions):

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-8

echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

nvcc --version   # MUST say "release 12.8"
```

⚠️ `nvidia-smi` will show `CUDA Version: 13.0` at the top (max supported by the driver) —
that's normal and irrelevant. The source of truth is `nvcc --version`.

## II.2 Python 3.10 (via deadsnakes)

```bash
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev

python3.10 --version   # Python 3.10.20
```

⚠️ Python 3.10 is mandatory: `tensorflow==2.15.0` has no wheel for 3.12+, and the other pins
(`tokenizers`, `timm`) are tested on 3.10. If `python3.10` disappears after a system update,
reinstall with `sudo apt install --reinstall python3.10 python3.10-venv python3.10-dev` and
verify `/usr/bin/python3.10` is owned by `root:root` with a real timestamp.

## II.3 CMake (to build LIBERO's egl_probe)

```bash
sudo apt install cmake
cmake --version
```

## II.4 VS Code native (.deb — NOT Flatpak)

The Flatpak build has filesystem sandboxing that breaks interpreter resolution inside venvs.

```bash
sudo apt install -y wget gpg
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" | sudo tee /etc/apt/sources.list.d/vscode.list
sudo apt update
sudo apt install -y code
rm packages.microsoft.gpg
```

## II.5 VRAM environment variable (mitigates marginal OOMs)

```bash
echo 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' >> ~/.bashrc
source ~/.bashrc
```

---

# PART III — OpenVLA ENVIRONMENT (`.openvla-env`)

## III.1 Venv + PyTorch

```bash
cd ~/Desktop/thesis-vla
python3.10 -m venv .openvla-env
source .openvla-env/bin/activate
which python && python --version   # MUST show .openvla-env/bin/python, 3.10.20

python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
# Expected: 2.11.0+cu128  True  (12, 0)
```

## III.2 Clone + install OpenVLA (ALWAYS --no-deps)

```bash
cd ~/Desktop/thesis-vla/vla-benchmark
git clone https://github.com/openvla/openvla.git
cd openvla
git rev-parse HEAD   # -> provenance

pip install -e . --no-deps
```

⚠️ Without `--no-deps`, pip would reinstall `torch==2.2.0` (pinned in the pyproject),
breaking Blackwell support.

## III.3 Manual dependencies (torch excluded)

```bash
pip install accelerate>=0.25.0 draccus==0.8.0 einops huggingface_hub json-numpy jsonlines \
    matplotlib peft==0.11.1 protobuf rich sentencepiece==0.1.99 timm==0.9.10 tokenizers==0.19.1 \
    transformers==4.40.1 wandb tensorflow==2.15.0 tensorflow_datasets==4.9.3 tensorflow_graphics==2021.12.3
pip install "dlimp @ git+https://github.com/moojink/dlimp_openvla"
```

## III.4 Fix protobuf / tensorflow-metadata

Symptom: `ImportError: cannot import name 'runtime_version' from 'google.protobuf'`

```bash
pip install "tensorflow-metadata==1.14.0" --force-reinstall --no-deps
```

## III.5 Fix accelerate (for quantization)

Symptom: `ValueError: .to is not supported for 4-bit or 8-bit bitsandbytes models`

```bash
pip install "accelerate==0.30.1" "torch==2.11.0+cu128" \
    --extra-index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0))"  # verify torch is intact
```

## III.6 bitsandbytes

```bash
pip install bitsandbytes
```

## III.7 Flash-attention — does NOT work on Blackwell, use SDPA

```bash
pip install packaging ninja
pip install flash-attn --no-build-isolation   # compiles but fails at runtime
```

Runtime symptom: `RuntimeError: FlashAttention only supports Ampere GPUs or newer`.
Don't uninstall it — just always use `attn_implementation="sdpa"` in code.

## III.8 Final check

```bash
python -c "
import torch; print('torch:', torch.__version__, '| cap:', torch.cuda.get_device_capability(0))
import transformers, tensorflow as tf, timm
print('transformers:', transformers.__version__, '| tf:', tf.__version__, '| timm:', timm.__version__)
"
deactivate
```

---

# PART IV — OpenVLA-OFT ENVIRONMENT (`.oft-env`)

## IV.1 Venv + PyTorch

```bash
cd ~/Desktop/thesis-vla
python3.10 -m venv .oft-env
source .oft-env/bin/activate
which python && python --version

python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0))"
```

## IV.2 Clone + install OFT

```bash
cd ~/Desktop/thesis-vla/vla-benchmark
git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
git rev-parse HEAD   # -> provenance

pip install -e . --no-deps
```

## IV.3 Manual dependencies (torch excluded)

```bash
pip install accelerate>=0.25.0 draccus==0.8.0 einops huggingface_hub json-numpy jsonlines \
    matplotlib peft==0.11.1 protobuf rich sentencepiece==0.1.99 timm==0.9.10 tokenizers==0.19.1 \
    wandb tensorflow==2.15.0 tensorflow_datasets==4.9.3 tensorflow_graphics==2021.12.3 \
    diffusers==0.30.3 uvicorn fastapi imageio
```

## IV.4 ⚠️ Transformers — MANDATORY FORK

OFT requires a custom fork with bidirectional attention for parallel decoding.

```bash
pip install "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git"
pip install "dlimp @ git+https://github.com/moojink/dlimp_openvla"
```

Look for `Resolved ...transformers-openvla-oft.git to commit <hash>` in the install log and
note it down. The reported version stays `4.40.1` (normal — it's a fork of that tag).

## IV.5 Fix protobuf + accelerate (same as OpenVLA)

```bash
pip install "tensorflow-metadata==1.14.0" --force-reinstall --no-deps
pip install "accelerate==0.30.1" "torch==2.11.0+cu128" \
    --extra-index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
```

## IV.6 bitsandbytes + flash-attn (same as OpenVLA)

```bash
pip install bitsandbytes
pip install packaging ninja
pip install flash-attn --no-build-isolation   # fails at runtime, ignore, use sdpa
```

## IV.7 Final check

```bash
python -c "
import torch; print('torch:', torch.__version__, '| cap:', torch.cuda.get_device_capability(0))
import transformers; print('transformers (fork):', transformers.__version__)
import tensorflow as tf, diffusers, dlimp
print('tf:', tf.__version__, '| diffusers:', diffusers.__version__, '| dlimp OK')
"
deactivate
```

---

# PART V — LIBERO (shared between both environments)

## V.1 Single shared clone

```bash
cd ~/Desktop/thesis-vla
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
git rev-parse HEAD   # -> provenance
```

⚠️ **Never `pip install -r requirements.txt`** (pins torch 1.11+cu113, would destroy the
Blackwell build).

## V.2 Full tested install sequence (for EACH venv)

Order matters: each package unblocks the next import error in the chain.

```bash
source ~/Desktop/thesis-vla/.openvla-env/bin/activate   # then repeat with .oft-env
cd ~/Desktop/thesis-vla/LIBERO

# 1. LIBERO itself, no dependencies
pip install -e . --no-deps

# 2. robosuite MUST be <1.5 (LIBERO uses SingleArmEnv, removed in 1.5)
pip install "robosuite==1.4.1"
pip install mujoco==3.9.0

# 3. gymnasium
pip install "gymnasium>=0.29.0"   # resolved to 1.3.0 in this environment

# 4. bddl MUST be ==1.0.1 (not the latest from PyPI)
pip install "bddl==1.0.1" --force-reinstall

# 5. robomimic (requires system CMake, see II.3)
pip install robomimic==0.2.0

# 6. egl_probe (HF fork: hf-egl-probe / hf_egl_probe)
pip install hf-egl-probe

# 7. CRITICAL FIX: bddl pulls in numpy>=2.0, which breaks tensorflow==2.15.0
pip install "numpy<2.0.0" --force-reinstall
```

Harmless expected warning: `libero 0.1.1 requires robosuite==1.4.0, but you have 1.4.1` —
tolerable.

**Versions actually resolved and verified in both venvs** (see `doc/pip-lists/`):
`robosuite==1.4.1`, `mujoco==3.9.0`, `gymnasium==1.3.0`, `bddl==1.0.1`, `robomimic==0.2.0`,
`egl_probe==1.0.2`, `hf_egl_probe==1.0.2`, `numpy==1.26.4`.

## V.3 Verification (4 checks)

```bash
python -c "from libero.libero import benchmark; print('LIBERO OK')"
python -c "from experiments.robot.libero.run_libero_eval import GenerateConfig; print('OFT+LIBERO OK')"
python -c "import tensorflow as tf; print('TF OK:', tf.__version__)"
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0))"
```

Expected: `LIBERO OK`, `OFT+LIBERO OK`, `TF OK: 2.15.0`, `2.11.0+cu128 (12, 0)`.
(Checks 2-3 require the transformers fork + OFT modules — they will fully pass only in
`.oft-env`.)

## V.4 Repeat V.2-V.3 in the second venv

Same exact sequence in `.oft-env` (same shared clone, separate editable install).

## V.5 Real end-to-end OFT test

```bash
cd ~/Desktop/thesis-vla/vla-benchmark/openvla-oft
python ../test_oft_libero.py --quant 8bit
```

Should produce an action chunk of shape `(8, 7)`, plausible small values, ~8-9 GB peak VRAM
at 8-bit.

---

# PART VI — PROVENANCE

> Fill in the `___` fields with your actual installation values. This section is the source
> of truth for the thesis' Methods section and for reproducibility.

## VI.1 Hardware / System

```
GPU:                  NVIDIA GeForce RTX 5080, 16 GB (Blackwell, compute capability sm_120)
Real available VRAM:  ~15.1 GB (16.3 GB nominal - ~1.1 GB COSMIC compositor overhead)
OS:                   Pop!_OS 24.04 LTS (Ubuntu 24.04 noble base)
NVIDIA driver:        580.159.03
CUDA (driver):        13.0 (max supported)
CUDA (toolkit):       12.8.93   <- the one actually used
Python:               3.10.20 (deadsnakes PPA)
```

## VI.2 Key shared versions (both venvs)

```
torch:               2.11.0+cu128   (DEVIATION: repos pin 2.2.0, incompatible with sm_120)
capability:          (12, 0)
transformers:        4.40.1  (OpenVLA: standard PyPI | OFT: FORK moojink/transformers-openvla-oft)
accelerate:          0.30.1  (DEVIATION: pinned for the quantization fix, was 1.14.0)
tensorflow:          2.15.0
tensorflow-metadata: 1.14.0  (DEVIATION: downgraded to fix the protobuf runtime_version issue)
timm:                0.9.10
tokenizers:          0.19.1
numpy:               1.26.4 (<2.0.0, forced for tensorflow compatibility after installing LIBERO)
bitsandbytes:        0.49.2
robosuite:           1.4.1
mujoco:              3.9.0
gymnasium:           1.3.0
bddl:                1.0.1
robomimic:           0.2.0
egl_probe / hf_egl_probe: 1.0.2
```

## VI.3 Commit hashes

```
openvla (openvla/openvla):                       c8f03f48af692657d3060c19588038c7220e9af9
openvla-oft (moojink/openvla-oft):                e4287e94541f459edc4feabc4e181f537cd569a8
transformers fork (transformers-openvla-oft):    ___ (annotate from install log, see IV.4)
LIBERO (Lifelong-Robot-Learning/LIBERO):         8f1084e3132a39270c3a13ebe37270a43ece2a01
```

## VI.4 Checkpoints used (HuggingFace)

```
OpenVLA base:  openvla/openvla-7b  (unnorm_key: bridge_orig)
OFT LIBERO:    moojink/openvla-7b-oft-finetuned-libero-spatial  (unnorm_key: libero_spatial_no_noops)
               moojink/openvla-7b-oft-finetuned-libero-object   (unnorm_key: libero_object_no_noops)
               moojink/openvla-7b-oft-finetuned-libero-goal     (unnorm_key: libero_goal_no_noops)
               moojink/openvla-7b-oft-finetuned-libero-10       (unnorm_key: libero_10_no_noops)
OpenVLA base LIBERO-finetuned: openvla/openvla-7b-finetuned-libero-{spatial,object,goal,10}
               (unnorm_key: libero_{spatial,object,goal,10} — NOTE: no _no_noops suffix,
               unlike the OFT checkpoints above; see core/policy_wrapper.py / config/loader.py
               unnorm_key resolution fallback)
```

## VI.5 Deviations from the official setup (to cite in the thesis Methods section)

```
1. torch 2.11.0+cu128 instead of 2.2.0
   Reason: RTX 5080 Blackwell (sm_120) requires CUDA >=12.8; torch 2.2.0 has no sm_120 kernels.
   Impact: hardware constraint, not a methodological choice.

2. attn_implementation="sdpa" instead of "flash_attention_2"
   Reason: flash-attn 2.x rejects sm_120 at runtime (hardcoded "Ampere or newer" check).
   Impact: SDPA/cuDNN nearly closes the performance gap on Blackwell (~160 TFLOPS).

3. 8-bit/4-bit quantization as a mandatory experimental axis (not optional)
   Reason: on 16GB with an active desktop compositor, the bf16 model (~14.2GB) risks OOM.
   Impact: quantization becomes an independent variable of the study, not a fallback.
   Measured VRAM: OpenVLA base bf16 ~14.2GB | 8-bit ~7.85GB | OFT 8-bit ~8.78GB

4. accelerate 0.30.1 and tensorflow-metadata 1.14.0 (downgrades)
   Reason: compatibility conflicts with transformers==4.40.1 (dispatch_model for
   quantization, and protobuf runtime_version, respectively). Do not alter model behavior.
```

## VI.6 Bugs encountered (issue references)

```
- FlashAttention doesn't support sm_120:              Dao-AILab/flash-attention#2361, #1665
- nvcc segfault building flash-attn for sm_120:        Dao-AILab/flash-attention#2361
- robosuite 1.5 removes SingleArmEnv:                  Lifelong-Robot-Learning/LIBERO#49,
                                                        ARISE-Initiative/robosuite#560
- mamba-ssm cublasLt crash on CUDA 12.9:               state-spaces/mamba#815   (relevant to future MaIL)
- Fix for mamba-ssm on RTX 50 series (setup patch):    state-spaces/mamba#745   (relevant to future MaIL)
- mujoco>=3.10.0 breaks robosuite==1.4.1's OSC controller (mj_fullM signature changed from
  (m, dst, M_sparse) to (m, d, dst)): fixed by pinning mujoco==3.9.0, the newest version
  with the old signature (see .claude/PROGRESS.md).
```

---

# PART VII — FINAL CHECKLIST

```
System:
[ ] nvcc --version -> 12.8.xx
[ ] nvidia-smi -> RTX 5080, driver >= 570
[ ] python3.10 --version -> 3.10.20
[ ] cmake --version works
[ ] code --version works (native, not Flatpak)

.openvla-env:
[ ] torch 2.11.0+cu128, capability (12,0)
[ ] transformers 4.40.1 (standard), tensorflow 2.15.0, timm 0.9.10
[ ] accelerate 0.30.1, tensorflow-metadata 1.14.0, numpy <2.0.0
[ ] bitsandbytes OK
[ ] LIBERO importable (robosuite 1.4.1, mujoco 3.9.0, gymnasium 1.3.0, bddl 1.0.1,
    robomimic 0.2.0, egl_probe/hf_egl_probe 1.0.2)
[ ] run_openvla_inference.py --quant 8bit works

.oft-env:
[ ] torch 2.11.0+cu128, capability (12,0)
[ ] transformers FORK (commit noted), diffusers 0.30.3, dlimp OK
[ ] LIBERO importable (same package set as .openvla-env) + GenerateConfig importable
[ ] test_oft_libero.py --quant 8bit -> action chunk (8,7), VRAM ~8-9GB

.mail-env (not yet built):
[ ] TODO — separate Python/CUDA/mamba-ssm stack, see mail/requirements.txt and
    doc/pip-lists/.mail-env.pip.md (to be created)

Provenance:
[ ] all commit hashes filled in (VI.3)
[ ] bitsandbytes version annotated (VI.2)
```

---

# NOTE — VRAM on native Linux desktop

On native Pop!\_OS, the COSMIC compositor + open windows use ~1-1.5GB of VRAM before Python
even starts. Out of a nominal 16.3GB, the real budget is ~15GB. The bf16 model (~14.2GB)
risks OOM. This differs from WSL2/headless servers where the GPU isn't shared with a
graphical compositor. Mitigations: (1) close graphical apps, (2) use 8/4-bit quantization,
(3) run from a TTY with the display manager stopped. Cite this in the thesis as a practical
constraint of deploying on consumer hardware.
