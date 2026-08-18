# vla-benchmark

Benchmark harness for evaluating Transformer-based and SSM-based VLA / MaIL policies on LIBERO.

## Prerequisites and official Setup

Three packages are **hardware/CUDA-version specific** and must be installed by hand
*before* `pip install .`: **torch**, **causal-conv1d**, and **mamba-ssm** (which needs
its own separately-compiled `selective_scan_cuda` kernel). They're intentionally left
out of `pyproject.toml` since the correct build depends on your GPU's compute
capability, driver, and CUDA toolkit version, and pip can't pick that for you.
Everything else needed is declared in `pyproject.toml`.

Setup used on an RTX 5080 (Blackwell, sm_120), driver CUDA 13.0 / nvcc 12.8:
- Python 3.10.20
- torch 2.11.0+cu128
- causal-conv1d 1.6.2.post1
- mamba-ssm 2.3.2.post1

No prebuilt wheels existed yet for that combination, so causal-conv1d and mamba-ssm
were compiled from source (see below). Expect the same on any very recent GPU
architecture — older/more common GPUs are more likely to find a matching wheel.

### 0. Clone the sub-projects

This repo doesn't vendor `mail`, `openvla`, `openvla-oft`, `LIBERO`, or `roboMamba-libero`
as submodules — clone each one into `external/` by hand before continuing (see
`config/paths.py` for the exact expected layout):

```bash
mkdir -p external
git clone git@github.com:ALRhub/MaIL.git external/mail
git clone git@github.com:openvla/openvla.git external/openvla
git clone git@github.com:moojink/openvla-oft.git external/openvla-oft
git clone git@github.com:Lifelong-Robot-Learning/LIBERO.git external/LIBERO
git clone --branch libero-port --single-branch git@github.com:danieleCapuzzo/roboMamba-libero.git external/roboMamba-libero
```

RoboMamba training/fine-tuning lives entirely in `roboMamba-libero` (a separate project);
this repo only loads the merged checkpoints it produces (see `doc/ROBOMAMBA.md`).

### 1. Create the environment

```bash
conda create -n bench-env python=3.10 -y
conda activate bench-env
```

### 2. Install torch for your CUDA setup

Check your driver's CUDA version first:

```bash
nvidia-smi      # see top-right corner
nvcc --version  # CUDA toolkit version
```

Install torch from the build matching your driver's CUDA version, using the selector
at https://pytorch.org/get-started/locally/. For example, for CUDA 12.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify it can see your GPU before continuing:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If `torch.cuda.is_available()` is `False`, stop and fix that first: causal-conv1d and
mamba-ssm both build CUDA extensions *against this exact torch install*, so a CPU-only
or mismatched torch will make the rest fail in confusing ways.

Note the GPU's compute capability while you're here
(`torch.cuda.get_device_capability(0)`, or look up your card) — you'll need it if you
end up compiling from source.

### 3. Install causal-conv1d

Try a prebuilt wheel first:

```bash
pip install causal-conv1d --no-deps --no-build-isolation
```

`pip` will attempt to build it locally using whatever's on your system (fast if a
matching prebuilt wheel is fetched from the project's GitHub releases during the
build, slow if it falls through to a full source compile). **Always pass `--no-deps`**
to avoid installing other torch versions by mistake. Check the compiled extension
actually loaded:

```bash
python -c "import causal_conv1d_cuda; print('ok')"
```

If that import fails, check the official package repo for instructions.

### 4. Install mamba-ssm

```bash
MAMBA_KEEP_CUDA_BUILD=TRUE pip install mamba-ssm --no-build-isolation
```
This may install another torch version. If so, force-reinstall the one you set up in
step 2. Verify the compiled kernel loaded (this is the part that actually matters, not
just `import mamba_ssm`):
```bash
python -c "import selective_scan_cuda; print('ok')"
```
If it fails, compile from source.

### 5. Install everything else

From the repo root, with `bench-env` active:

```bash
pip install -e .
```

Then install the cloned sub-projects as editable installs:

```bash
pip install -e ./external/openvla --no-deps
pip install -e ./external/openvla-oft --no-deps
```
*NOTE:* `external/roboMamba-libero/` and `external/mail/` do not need to be installed.

Finally, install `libero` from PyPI, also with `--no-deps`:

```bash
pip install libero==0.1.1 --no-deps
```

`--no-deps` avoids downgrading the `robomimic` version this project needs
and conflicts with in a single resolve — see the comment in `pyproject.toml` next to
the (deliberately absent) `libero` dependency. 

### 7. Verifying the install

Run the dedicated environment check first — it isolates failures in torch/CUDA,
causal-conv1d, and mamba-ssm from everything else, with failure messages that point
back at the section above:

```bash
python tests/test_env_setup.py
```

Then run the full test suite:

```bash
pytest
```

`tests/test_mail_env_smoke.py` and `tests/test_env_setup.py` are both skipped
automatically if no CUDA device is visible; everything else runs on CPU.

## Usage

### Evaluate a policy

`run_eval.py` wires the policy factory, wrapper, `LiberoEvaluator`, and results I/O
into one CLI. Camera count and render resolution are fixed per model (see
`config/constants.py` `NUM_CAMERAS_BY_MODEL` / `RENDER_RESOLUTION_BY_MODEL`), and the config
is auto-selected as `config/config_<model>.json` unless `--config` is given explicitly.

```bash
python run_eval.py --model oft --suite spatial --quant 8bit --rollouts 20
python run_eval.py --model openvla --suite all --quant 8bit       # full benchmark
python run_eval.py --model mail --suite goal --language           # language-conditioned MaIL
python run_eval.py --model robomamba --suite spatial              # RoboMamba (merged checkpoint)
```

Key flags: 
- `--model {openvla,oft,mail,robomamba}`, 
- `--suite {spatial,object,goal,long,all}` (accepts multiple), 
- `--quant {none,8bit,4bit}`, 
- `--rollouts N`, 
- `--seed N`,
- `--output-dir DIR`,
- `--save-videos` (dumps an MP4 per rollout), 
- `--language` (MaIL-only, natural-language task conditioning).

Results are written under `results/<model>_<quant>_<ncam>cam/<suite>[_language]/`.

### Train MaIL

`train_mail.py` trains MambaIL (DDPM, Enc-Dec) on a single LIBERO suite:

```bash
python train_mail.py --suite spatial
python train_mail.py --suite long --num-demos 10   # official paper 20%-data setting
```

Trains on 100% of demos per task by default, other important flags: 
- `--epochs`, 
- `--batch-size`,
- `--lr`, 
- `--ema-decay`, 
- `--seed`, 
- `--num-workers`, 
- `--amp` (bf16 autocast),
- `--output-dir` (default `trained/`).

### RoboMamba checkpoints

RoboMamba has no training entrypoint in this repo — fine-tuning happens in the separate
`roboMamba-libero` project (`external/roboMamba-libero/`), which produces a single
self-contained merged `.pt` per suite via its `merge_weights.py`. Copy that file into
`trained/robomamba/<suite>/` and point `config/config_robomamba.json` at it; see
`doc/ROBOMAMBA.md` for the checkpoint format and how it's loaded.
