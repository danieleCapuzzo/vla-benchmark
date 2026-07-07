# Setup Ambienti — Tesi VLA su LIBERO

### Guida completa di ricostruzione + Provenance — RTX 5080 (Blackwell sm_120), Pop!\_OS 24.04

> Documento unico e auto-contenuto. Copre: prerequisiti sistema, i due ambienti Python
> (`.openvla-env`, `.oft-env`), LIBERO condiviso, struttura cartelle, e la provenance completa
> (versioni, commit, deviazioni, bug incontrati).
> Ultima verifica end-to-end: 5 luglio 2026 — inferenza OFT reale su LIBERO-Spatial funzionante.
>
> **Ordine obbligatorio:** Sistema → Python → OpenVLA → OFT → LIBERO (in entrambi i venv).

---

# PARTE I — STRUTTURA CARTELLE

```
~/Desktop/thesis-vla/
├── .openvla-env/                 # venv OpenVLA base (Python 3.10)
├── .oft-env/                      # venv OpenVLA-OFT (Python 3.10)
├── LIBERO/                        # clone UNICO condiviso, installato editable in ENTRAMBI i venv
└── vla-benchmark/                 # repo Git di lavoro
    ├── openvla/                    # clone repo ufficiale openvla/openvla
    ├── openvla-oft/                # clone repo ufficiale moojink/openvla-oft
    ├── provenance.md               # log di provenance (compilare con gli hash reali)
    ├── run_openvla_inference.py    # inferenza OpenVLA base, quant selezionabile
    ├── test_oft_libero.py          # test/benchmark OFT via harness ufficiale
    └── .vscode/
        └── settings.json           # interprete puntato al venv corretto
```

**Note sulla struttura:**

- `.openvla-env` e `.oft-env` stanno in `thesis-vla/` (un livello sopra `vla-benchmark/`).
- `LIBERO/` è clonato una sola volta e installato come editable (`pip install -e .`) in
  entrambi i venv: un solo sorgente, un solo commit da tracciare.
- I repo `openvla/` e `openvla-oft/` stanno DENTRO `vla-benchmark/`. Aggiungi al `.gitignore`
  di `vla-benchmark/`: `openvla/`, `openvla-oft/`, `LIBERO/`, `__pycache__/`, `*.pyc` — sono
  repo Git separati con la loro storia, tu tieni traccia solo dei commit hash nel provenance.

**VS Code `settings.json`** (in `vla-benchmark/.vscode/`, per lo scenario "apri vla-benchmark come root"):

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

Per lavorare su OFT, cambia i due path in `.oft-env`. In alternativa usa un multi-root
workspace con due `.vscode/settings.json` separati (uno per `openvla/`, uno per `openvla-oft/`).

---

# PARTE II — PREREQUISITI DI SISTEMA (una tantum)

## II.1 Driver NVIDIA + CUDA Toolkit 12.8

```bash
nvidia-smi   # verifica driver >= 570 e GPU RTX 5080 visibile
```

Installa CUDA Toolkit 12.8 (NON 12.9 né 13.0 — bug noti su Blackwell con quelle versioni):

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-8

echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

nvcc --version   # DEVE dire "release 12.8"
```

⚠️ `nvidia-smi` mostrerà in alto `CUDA Version: 13.0` (max supportato dal driver) — è normale
e ininfluente. La verità è `nvcc --version`.

## II.2 Python 3.10 (via deadsnakes)

```bash
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev

python3.10 --version   # Python 3.10.20
```

⚠️ Python 3.10 obbligatorio: `tensorflow==2.15.0` non ha wheel per 3.12+, e le altre pin
(`tokenizers`, `timm`) sono testate su 3.10. Se `python3.10` sparisce dopo un aggiornamento
di sistema, reinstalla con `sudo apt install --reinstall python3.10 python3.10-venv python3.10-dev`
e verifica che `/usr/bin/python3.10` abbia owner `root:root` e data reale.

## II.3 CMake (per compilare egl_probe di LIBERO)

```bash
sudo apt install cmake
cmake --version
```

## II.4 VS Code nativo (.deb — NON Flatpak)

Il Flatpak ha sandboxing filesystem che rompe la risoluzione degli interpreti nei venv.

```bash
sudo apt install -y wget gpg
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" | sudo tee /etc/apt/sources.list.d/vscode.list
sudo apt update
sudo apt install -y code
rm packages.microsoft.gpg
```

## II.5 Variabile d'ambiente per VRAM (mitiga OOM marginali)

```bash
echo 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' >> ~/.bashrc
source ~/.bashrc
```

---

# PARTE III — AMBIENTE OpenVLA (`.openvla-env`)

## III.1 Venv + PyTorch

```bash
cd ~/Desktop/thesis-vla
python3.10 -m venv .openvla-env
source .openvla-env/bin/activate
which python && python --version   # DEVE mostrare .openvla-env/bin/python, 3.10.20

python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
# Atteso: 2.11.0+cu128  True  (12, 0)
```

## III.2 Clone + install OpenVLA (SEMPRE --no-deps)

```bash
cd ~/Desktop/thesis-vla/vla-benchmark
git clone https://github.com/openvla/openvla.git
cd openvla
git rev-parse HEAD   # -> provenance

pip install -e . --no-deps
```

⚠️ Senza `--no-deps`, pip reinstallerebbe `torch==2.2.0` (pinnato nel pyproject), rompendo Blackwell.

## III.3 Dipendenze manuali (escluso torch)

```bash
pip install accelerate>=0.25.0 draccus==0.8.0 einops huggingface_hub json-numpy jsonlines \
    matplotlib peft==0.11.1 protobuf rich sentencepiece==0.1.99 timm==0.9.10 tokenizers==0.19.1 \
    transformers==4.40.1 wandb tensorflow==2.15.0 tensorflow_datasets==4.9.3 tensorflow_graphics==2021.12.3
pip install "dlimp @ git+https://github.com/moojink/dlimp_openvla"
```

## III.4 Fix protobuf / tensorflow-metadata

Sintomo: `ImportError: cannot import name 'runtime_version' from 'google.protobuf'`

```bash
pip install "tensorflow-metadata==1.14.0" --force-reinstall --no-deps
```

## III.5 Fix accelerate (per quantizzazione)

Sintomo: `ValueError: .to is not supported for 4-bit or 8-bit bitsandbytes models`

```bash
pip install "accelerate==0.30.1" "torch==2.11.0+cu128" \
    --extra-index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0))"  # verifica torch intatto
```

## III.6 bitsandbytes

```bash
pip install bitsandbytes
```

## III.7 Flash-attention — NON funziona su Blackwell, usa SDPA

```bash
pip install packaging ninja
pip install flash-attn --no-build-isolation   # compila ma fallisce a runtime
```

Sintomo runtime: `RuntimeError: FlashAttention only supports Ampere GPUs or newer`.
Non disinstallare, semplicemente usa sempre `attn_implementation="sdpa"` nel codice.

## III.8 Verifica finale

```bash
python -c "
import torch; print('torch:', torch.__version__, '| cap:', torch.cuda.get_device_capability(0))
import transformers, tensorflow as tf, timm
print('transformers:', transformers.__version__, '| tf:', tf.__version__, '| timm:', timm.__version__)
"
deactivate
```

---

# PARTE IV — AMBIENTE OpenVLA-OFT (`.oft-env`)

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

## IV.3 Dipendenze manuali (escluso torch)

```bash
pip install accelerate>=0.25.0 draccus==0.8.0 einops huggingface_hub json-numpy jsonlines \
    matplotlib peft==0.11.1 protobuf rich sentencepiece==0.1.99 timm==0.9.10 tokenizers==0.19.1 \
    wandb tensorflow==2.15.0 tensorflow_datasets==4.9.3 tensorflow_graphics==2021.12.3 \
    diffusers==0.30.3 uvicorn fastapi imageio
```

## IV.4 ⚠️ Transformers — FORK OBBLIGATORIA

OFT richiede una fork custom con attenzione bidirezionale per il parallel decoding.

```bash
pip install "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git"
pip install "dlimp @ git+https://github.com/moojink/dlimp_openvla"
```

Nel log di install cerca `Resolved ...transformers-openvla-oft.git to commit <hash>` e annota
l'hash. Il numero di versione mostrato resta `4.40.1` (normale, è una fork di quel tag).

## IV.5 Fix protobuf + accelerate (identici a OpenVLA)

```bash
pip install "tensorflow-metadata==1.14.0" --force-reinstall --no-deps
pip install "accelerate==0.30.1" "torch==2.11.0+cu128" \
    --extra-index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
```

## IV.6 bitsandbytes + flash-attn (stesso discorso di OpenVLA)

```bash
pip install bitsandbytes
pip install packaging ninja
pip install flash-attn --no-build-isolation   # fallisce a runtime, ignora, usa sdpa
```

## IV.7 Verifica finale

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

# PARTE V — LIBERO (condiviso tra i due ambienti)

## V.1 Clone unico condiviso

```bash
cd ~/Desktop/thesis-vla
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
git rev-parse HEAD   # -> provenance
```

⚠️ **Mai `pip install -r requirements.txt`** (pinna torch 1.11+cu113, distruggerebbe Blackwell).

## V.2 Sequenza di installazione completa e testata (per CIASCUN venv)

L'ordine conta: ogni pacchetto sblocca l'errore di import successivo nella catena.

```bash
source ~/Desktop/thesis-vla/.openvla-env/bin/activate   # poi ripeti con .oft-env
cd ~/Desktop/thesis-vla/LIBERO

# 1. LIBERO stesso, senza dipendenze
pip install -e . --no-deps

# 2. robosuite DEVE essere <1.5 (LIBERO usa SingleArmEnv, rimosso in 1.5)
pip install "robosuite==1.4.1"
pip install mujoco==3.9.0

# 3. gymnasium
pip install "gymnasium>=0.29.0"

# 4. bddl DEVE essere ==1.0.1 (non l'ultima da PyPI)
pip install "bddl==1.0.1" --force-reinstall

# 5. robomimic (richiede CMake di sistema, vedi II.3)
pip install robomimic==0.2.0

# 6. hf-egl-probe
pip install hf-egl-probe

# 7. FIX CRITICO: bddl tira dentro numpy>=2.0 che rompe tensorflow==2.15.0
pip install "numpy<2.0.0" --force-reinstall
```

Warning innocuo atteso: `libero 0.1.1 requires robosuite==1.4.0, but you have 1.4.1` — tollerabile.

## V.3 Verifica (4 controlli)

```bash
python -c "from libero.libero import benchmark; print('LIBERO OK')"
python -c "from experiments.robot.libero.run_libero_eval import GenerateConfig; print('OFT+LIBERO OK')"
python -c "import tensorflow as tf; print('TF OK:', tf.__version__)"
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0))"
```

Atteso: `LIBERO OK`, `OFT+LIBERO OK`, `TF OK: 2.15.0`, `2.11.0+cu128 (12, 0)`.
(I check 2-3 richiedono la fork transformers + moduli OFT: passeranno pienamente solo in `.oft-env`.)

## V.4 Ripeti V.2-V.3 nel secondo venv

Stessa identica sequenza in `.oft-env` (stesso clone condiviso, install editable separato).

## V.5 Test end-to-end OFT reale

```bash
cd ~/Desktop/thesis-vla/vla-benchmark/openvla-oft
python ../test_oft_libero.py --quant 8bit
```

Deve dare action chunk shape `(8, 7)`, valori piccoli plausibili, VRAM picco ~8-9 GB in 8-bit.

---

# PARTE VI — PROVENANCE

> Compila i campi `___` con i valori reali della tua installazione. Questa sezione è la fonte
> di verità per i Metodi della tesi e per la riproducibilità.

## VI.1 Hardware / Sistema

```
GPU:            NVIDIA GeForce RTX 5080, 16 GB (Blackwell, compute capability sm_120)
VRAM reale disponibile: ~15.1 GB (16.3 GB nominali - ~1.1 GB overhead compositor COSMIC)
OS:             Pop!_OS 24.04 LTS (base Ubuntu 24.04 noble)
Driver NVIDIA:  580.159.03
CUDA (driver):  13.0 (max supportato)
CUDA (toolkit): 12.8.93   <- quello effettivamente usato
Python:         3.10.20 (deadsnakes PPA)
```

## VI.2 Versioni chiave condivise (entrambi i venv)

```
torch:               2.11.0+cu128   (DEVIAZIONE: repo pinnano 2.2.0, incompatibile con sm_120)
capability:          (12, 0)
transformers:        4.40.1  (OpenVLA: standard PyPI | OFT: FORK moojink/transformers-openvla-oft)
accelerate:          0.30.1  (DEVIAZIONE: pinnato per fix quantizzazione, era 1.14.0)
tensorflow:          2.15.0
tensorflow-metadata: 1.14.0  (DEVIAZIONE: downgrade per fix protobuf runtime_version)
timm:                0.9.10
tokenizers:          0.19.1
numpy:               <2.0.0  (forzato per compatibilità tensorflow, dopo install LIBERO)
bitsandbytes:        ___ (versione installata)
```

## VI.3 Commit hash

```
openvla (openvla/openvla):                       c8f03f48af692657d3060c19588038c7220e9af9
openvla-oft (moojink/openvla-oft):               e4287e94541f459edc4feabc4e181f537cd569a8
transformers fork (transformers-openvla-oft)
LIBERO (Lifelong-Robot-Learning/LIBERO):         8f1084e3132a39270c3a13ebe37270a43ece2a01
```

## VI.4 Checkpoint usati (HuggingFace)

```
OpenVLA base:  openvla/openvla-7b  (unnorm_key: bridge_orig)
OFT LIBERO:    moojink/openvla-7b-oft-finetuned-libero-spatial  (unnorm_key: libero_spatial_no_noops)
               moojink/openvla-7b-oft-finetuned-libero-object   (unnorm_key: libero_object_no_noops)
               moojink/openvla-7b-oft-finetuned-libero-goal     (unnorm_key: libero_goal_no_noops)
               moojink/openvla-7b-oft-finetuned-libero-10       (unnorm_key: libero_10_no_noops)
```

## VI.5 Deviazioni dal setup ufficiale (da citare nei Metodi della tesi)

```
1. torch 2.11.0+cu128 invece di 2.2.0
   Motivo: RTX 5080 Blackwell (sm_120) richiede CUDA >=12.8; torch 2.2.0 non ha kernel sm_120.
   Impatto: vincolo hardware, non scelta metodologica.

2. attn_implementation="sdpa" invece di "flash_attention_2"
   Motivo: flash-attn 2.x rifiuta sm_120 a runtime (check hardcoded "Ampere or newer").
   Impatto: SDPA/cuDNN colma quasi del tutto il gap prestazionale su Blackwell (~160 TFLOPS).

3. Quantizzazione 8-bit/4-bit come asse sperimentale obbligato (non opzionale)
   Motivo: su 16GB con desktop grafico attivo, il modello bf16 (~14.2GB) rischia OOM.
   Impatto: la quantizzazione diventa variabile indipendente dello studio, non ripiego.
   VRAM misurate: OpenVLA base bf16 ~14.2GB | 8-bit ~7.85GB | OFT 8-bit ~8.78GB

4. accelerate 0.30.1 e tensorflow-metadata 1.14.0 (downgrade)
   Motivo: conflitti di compatibilità con transformers==4.40.1 (rispettivamente dispatch_model
   per quantizzazione, e protobuf runtime_version). Non alterano il comportamento dei modelli.
```

## VI.6 Bug incontrati (riferimenti issue)

```
- FlashAttention non supporta sm_120:         Dao-AILab/flash-attention#2361, #1665
- nvcc segfault flash-attn per sm_120:        Dao-AILab/flash-attention#2361
- robosuite 1.5 rimuove SingleArmEnv:         Lifelong-Robot-Learning/LIBERO#49,
                                              ARISE-Initiative/robosuite#560
- mamba-ssm cublasLt crash su CUDA 12.9:      state-spaces/mamba#815   (rilevante per MaIL futuro)
- Fix mamba-ssm RTX 50 series (patch setup):  state-spaces/mamba#745   (rilevante per MaIL futuro)
```

---

# PARTE VII — CHECKLIST FINALE

```
Sistema:
[ ] nvcc --version -> 12.8.xx
[ ] nvidia-smi -> RTX 5080, driver >= 570
[ ] python3.10 --version -> 3.10.20
[ ] cmake --version funziona
[ ] code --version funziona (nativo, non Flatpak)

.openvla-env:
[ ] torch 2.11.0+cu128, capability (12,0)
[ ] transformers 4.40.1 (standard), tensorflow 2.15.0, timm 0.9.10
[ ] accelerate 0.30.1, tensorflow-metadata 1.14.0, numpy <2.0.0
[ ] bitsandbytes OK
[ ] LIBERO importabile
[ ] run_openvla_inference.py --quant 8bit funziona

.oft-env:
[ ] torch 2.11.0+cu128, capability (12,0)
[ ] transformers FORK (commit annotato), diffusers 0.30.3, dlimp OK
[ ] robosuite <1.5, gymnasium, bddl==1.0.1, robomimic, hf-egl-probe, numpy <2.0.0
[ ] LIBERO importabile + GenerateConfig importabile
[ ] test_oft_libero.py --quant 8bit -> action chunk (8,7), VRAM ~8-9GB

Provenance:
[ ] tutti i commit hash compilati (VI.3)
[ ] versione bitsandbytes annotata (VI.2)
```

---

# NOTE — VRAM su desktop Linux nativo

Su Pop!\_OS nativo il compositor COSMIC + finestre aperte occupano ~1-1.5GB di VRAM prima di
lanciare Python. Su 16.3GB nominali il budget reale è ~15GB. Il modello bf16 (~14.2GB) rischia
OOM. Diverso da WSL2/server headless dove la GPU non è condivisa con un compositor grafico.
Soluzioni: (1) chiudere app grafiche, (2) quantizzazione 8/4-bit, (3) TTY con display manager
fermato. Da citare in tesi come vincolo pratico del deployment su hardware consumer.
