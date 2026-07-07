# CLAUDE.md — LIBERO VLA TESTBENCH

## What we are building

A **modular testbench** comparing three Vision-Language-Action models on LIBERO —
**OpenVLA**, **OpenVLA-OFT**, **MaIL** (Mamba imitation learning) — focused on **real-time
metrics** (latency, throughput, VRAM) alongside success rate and trajectory smoothness.

Read if requested by user: `doc/ARCHITECTURE.md` (design contract).
Read only if asked: `doc/SETUP_ENV.md` (environment build instructions).

## Package versions — read before using any model API

Exact pinned versions per environment live in `doc/pip-lists/`:

- `.openvla-env.pip.md` — OpenVLA base environment
- `.oft-env.pip.md` — OpenVLA-OFT environment (transformers is a FORK, runs openvla-oft ONLY)
- `.mail-env.pip.md` — MaIL environment (mamba-ssm, different Python/CUDA) — not yet created

Never assume standard versions — e.g. `transformers==4.40.1`, `accelerate==0.30.1`,
`numpy<2.0.0`, `robosuite==1.4.1` (must be `<1.5`), `torch==2.11.0+cu128` (Blackwell build).

## Available hardware — NON-NEGOTIABLE

- **GPU: RTX 5080, 16 GB (Blackwell, sm_120).** Real VRAM budget ~15 GB (compositor overhead).
- **flash-attn does not work** on Blackwell — always use `attn_implementation="sdpa"`.
- **Quantization**: models must be loadable in `none`/`8bit`/`4bit` as in the original papers.
  Quantized models use `device_map="auto"`, NEVER `.to("cuda")`; `pixel_values` must be
  `float16` (not `bfloat16`) whenever quantized.
- **torch 2.11.0+cu128**: never overwrite it (`--no-deps` on any install that touches it).

## Design rules (see ARCHITECTURE.md for the full contract)

- Do not duplicate official harnesses — import them as libraries and wrap them. If wrapping
  cleanly would mean copying substantial official logic to avoid spaghetti code, ask the
  user first.
- Each model implements the abstract `PolicyWrapper` interface (Strategy pattern).
- `LiberoEvaluator` is the Facade; `PolicyFactory` instantiates wrappers; the evaluation loop
  is identical for all models.
- Adding a model = new wrapper + factory registration, no changes to core.

## Evaluation protocol (FROZEN)

- 20 rollouts/task (200/suite), identical tasks and initial states across all models.
- 4 suites: Spatial, Object, Goal, Long.
- Cameras: 1 (agentview) primary; 2 optional for models that support it (OpenVLA-OFT, MaIL).
- Real-time metrics: latency (mean/p95/p99), throughput (actions/s, frequency), peak VRAM,
  time-to-first-action.
- Output per run: `episodes.jsonl` + `summary.csv` (+ `provenance.json`, not yet implemented).

## Repo layout

```
run_eval.py         CLI entrypoint (repo root, works from any cwd)
core/               PolicyWrapper contract, PolicyFactory, LiberoEvaluator, shared LIBERO utils
wrappers/           one adapter per model (openvla_wrapper.py, oft_wrapper.py, mail_wrapper.py)
                    + wrapper_utils.py shared by the OpenVLA-family wrappers
config/             config_openvla.json / config_oft.json (checkpoints, unnorm_keys) + loader.py
results_io/         episodes.jsonl / summary.csv writers
results/            run outputs (one dir per run)
tests/              pytest, mocked transformers/prismatic — no GPU/network required
doc/                ARCHITECTURE.md, SETUP_ENV.md, pip-lists/

openvla/            official repos — READ-ONLY, see below
openvla-oft/
LIBERO/
mail/
```

## Environments and how to run

A run = one model. Activate the matching venv before launching:

- `.oft-env` → base OpenVLA **and** OpenVLA-OFT
- `.mail-env` → MaIL (not yet built)

```bash
# example (assumes the correct venv is already active)
python run_eval.py --model oft --suite spatial --quant 8bit --num-cameras 1 --rollouts 20
```

## Code conventions

- Python 3.10. Mandatory type hints on all public interfaces.
- Timing: always `torch.cuda.synchronize()` before/after latency measurement; discard
  warm-up runs.
- Latency must measure only the model call itself, never preprocessing.
- Do not introduce dependencies without checking compatibility against `doc/pip-lists/`.
- Report environment/import errors to the user — do not silently work around them.

## What NOT to do

- Do not modify `openvla/`, `openvla-oft/`, `LIBERO/`, `mail/` (official read-only repos).
- Do not use `flash_attention_2`.
- Never override the pinned torch version (`--no-deps` where needed).
- Never assume package versions — consult `doc/pip-lists/`.
