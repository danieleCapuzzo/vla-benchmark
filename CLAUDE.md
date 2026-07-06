# CLAUDE.md — LIBERO VLA TESTBENCH

## What we are building

A **modular testbench** to compare three Vision-Language-Action Models su LIBERO, 
focusing on **real-time metrics** (latency, throughput, VRAM) and success rate, trajectory smoothness.
Models: **OpenVLA**, **OpenVLA-OFT**, **MaIL** (Mamba imitation learning).

A full architectural specification is available at: `doc/ARCHITECTURE.md` (read it before coding).
For environment setup refer to: `doc/SETUP_ENV.md` but read it only if asked.

## Package versions — Read before coding
Exact package versiona of every environment are reported in dedicated files located at `doc/pip-lists/...`
** Refer to that file before using specific APIs as versions are pinned for compatibility purposes**:

- `pip-lists/openvla-env.pip.md`  OpenVLA base environment
- `pip-lists/oft-env.pip.md`      OpenVLA-OFT environment (transformers è una FORK, works for openvla as well)
- `pip-lists/mail-env.pip.md`     MaIL environment (mamba-ssm, Python/CUDA diversi)

Never assume standard vesions, consider those in the files such as `transformers==4.40.1`, `accelerate==0.30.1`
, `numpy<2.0.0`, `robosuite<1.5`, `torch==2.11.0+cu128` (build Blackwell).

## Available hardware — NON NEGOTIABLE 
- **GPU: RTX 5080, 16 GB (Blackwell, sm_120).** real VRAM budget ~15 GB (overhead compositor).
- **flash-attn does not work** on Blackwell, so use `attn_implementation="sdpa"`.
- **Quantization**: models should be loadable in `none`/`8bit`/`4bit` as in the original papers.
  for quantized models use `device_map="auto"`, NEVER `.to("cuda")`; `pixel_values` should be
  `float16` (non bfloat16) for full precision vision backbone.
- **torch 2.11.0+cu128**: never overwrite it (use `--no-deps` on install).

## Design rules (read ARCHITECTURE.md for the full contract)
- **Do not duplicate** official harnesses, import them as libraries and wrap them.
- Each model should implement the abstract interface `PolicyWrapper` (Strategy pattern).
- `LiberoEvaluator` is the Facade; `PolicyFactory` instantiates wrappers; evaluation loop must be identical for all models.
- Adding a model should be as adding a new wrapper + registering it on the factory with no core modification.

## Evaluation protocol (FROZEN)

- **20 rollout/task** (200/suite), same tasks and initial states for all models.
- 4 suite: Spatial, Object, Goal, Long.
- Camere: **1 (agentview) primary**; optional support for 2 if the model can handle them (OpenVLA-OFT and MambaIL do).
- Real-time metrics: latency (mean/p95/p99), throughput (actions/s, frequency ...) , Peak VRAM, time-to-first-action.
- Output: `episodes.jsonl` + `summary.csv` + `provenance.json` for each run.

## Environments and how to run

A run = one model. User activates the correct venv before launching. Available aliases are 
- `oft-env` → OpenVLA base E OpenVLA-OFT
- `mail-env` → MaIL
to activate the correct venv

```bash
# example (assume user already activated the correct venv)
python testbench/run_eval.py --model oft --suite spatial --quant 8bit --num-cameras 1 --rollouts 20
```

## Code Conventions

- Python 3.10 (VLA) / see pip-list for MaIL. Mandatory Type hints on all public interfaces.
- Timing: always use `torch.cuda.synchronize()` before/after latency measures, discard warm-up runs.
- Each latency measure must consider only the time required for the model call, no preprocessing.
- Do not introduce dependencies without checking compatibility.
- Environment errors or import must be reported to user, do not fix randomly.

## What NOT to do
- Do not modify files in `openvla/`, `openvla-oft/`, `LIBERO/`, `MaIL/` (official read-only repositories).
- Do not use `flash_attention_2`.
- Do not ever override torch version (`--no-deps` where needed).
- Never assume package versions, consult `pip-lists/`.
