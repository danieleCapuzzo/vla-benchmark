# VLA Testbench Architecture on LIBERO
### Unified Evaluation System for OpenVLA, OpenVLA-OFT, and MaIL

> Design document. Defines the evaluation system architecture, 
> the decisions made, and the contract(interface) that each model must respect. 
> Serves as the foundational prompt for Claude Code.
---

## 1. Scientific Objective

A **speed–accuracy** tradeoff comparison between three policy paradigms on LIBERO, constrained to consumer GPUs (RTX 5080, 16 GB). The primary output is a **Pareto frontier** of success rate vs. real-time metrics (latency, throughput, control frequency), with VRAM as the third dimension.  

The primary scientific focus is on **real-time metrics**; the success rate serves as the "quality" axis of the tradeoff, not the main contribution.
---

## 2. Protocol Decisions (FROZEN)

| Decision | Value | Rationale |
| :---- | :---- | :---- |
| Rollouts per task | **20** (200/suite) | Official MaIL protocol; sustainable within a 3-week timeframe; focus is on real-time. |
| Suites | Spatial, Object, Goal, Long | 4 suites common to all three papers (LIBERO-90 excluded). |
| Tasks and initial states | **Identical for all 3** | Fair comparison: exact same environment init seeds. |
| Seeds | 3 (where applicable) | MaIL: training+eval; VLA: eval-time. |
| Cameras (primary) | **1 (agentview)** | 2-camera OFT does not fit reliably within 16GB. |
| Cameras (optional) | 2 for OFT | Supported parametrically, ablation if VRAM permits. |
| Quantization | none / 8bit / 4bit | As in original papers; necessary experimental axis on 16GB. |
| MaIL Training | 20% demo (primary) | Official script; 100% as an optional ablation. |
| Attention backend | sdpa | flash-attn is currently broken on Blackwell. |

**Known divergences from the original papers (to be declared in the Methods section):**

* OpenVLA and OFT use 50 rollouts/task in their papers; here we use 20 for consistency with MaIL and due to time constraints.  
* OFT uses 2 cameras \+ proprioception in the paper; here 1 camera is the primary setup due to VRAM limits.  
* All models run on Blackwell 5080 16GB VRAM hardware with possible quantization, not A100 bf16 as in the papers.
---


## 3. Cross-Environment Orchestration

**Execution model: one launch = one model.** The user manually activates the correct virtual environment before launching. There is no cross-venv subprocess orchestrator (an explicit choice for simplicity and robustness).

```
.oft-env    -> OpenVLA base AND OpenVLA-OFT (share prismatic/transformers-fork lineage)  
.mail-env   -> MaIL                         (mamba-ssm, potentially different Python/CUDA)
```

Note: `.openvla-env` and `.oft-env` are almost identical; the base OpenVLA runs fine in `.oft-env` (verified). To reduce the maintenance surface, `.oft-env` can be used for both VLAs (NOT for MambaIL).

Each run produces JSON/CSV files containing `model_id` and `env_id` in the metadata, allowing the final aggregator to combine results produced by separate launches in different environments.
---

## 4. Software Architecture (Design Patterns)
### 4.1 Principle: Minimal code wrapping official harnesses

The testbench **does not duplicate** the evaluation code from the official repositories. It imports them as libraries and wraps them behind a common interface. This guarantees that results remain faithful to the official harnesses (comparability with the papers) and minimizes the code to maintain.

### 4.2 Applied Patterns
**Strategy \+ Template Method (`PolicyWrapper` interface):**  
Each model is a concrete Strategy implementing a common abstract interface. The evaluation loop (Template Method) is identical for all; only the model-specific steps vary.
**Adapter:**  
Each wrapper adapts the model's native API (OpenVLA `predict_action`, OFT `get_vla_action`,
MaIL's own API) to the testbench's common interface.
**Facade:**  
A `LiberoEvaluator` class acts as a facade: it hides the complexity of LIBERO environment setup, rollout loops, metrics collection, and serialization. The user/Claude Code interacts only with this class.
**Factory:**  
A `PolicyFactory` instantiates the correct wrapper based on `--model {openvla,oft,mail}`. Adding a new model means adding a Strategy and registering it in the Factory (Open/Closed Principle).

### 4.3 Common Interface (The Contract)

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class PolicyOutput:
    """Output normalizzato di una singola chiamata di inferenza."""
    actions: np.ndarray          # shape (K, action_dim); K=1 for closed-loop, K>1 for chunked
    is_chunked: bool             # True of model returns chunked actions
    raw_latency_s: float         # latency of only model call (measured by wrapper)

@dataclass
class PolicyMetadata:
    """Metadati statici del modello, per provenance e tabelle."""
    model_id: str                # "openvla" | "oft" | "mail"
    variant: str                 # e.g., specific checkpoint or config
    num_params: int
    quant: str                   # "none" | "8bit" | "4bit"
    num_cameras: int
    action_chunk_size: int       # nominal K (1 for OpenVLA/MaIL closed-loop, 8 for OFT)
    attn_impl: str               # "sdpa"
    extra: dict                  # extra field for specific model details

class PolicyWrapper(ABC):  
    """Common interface that every model MUST implement (Strategy)."""

    @abstractmethod  
    def load(self, quant: str = "none", num_cameras: int = 1, **kwargs) -> None:  
        """Loads weights/checkpoints with the requested quantization and camera config.  
        Must use EXACTLY the procedures from the original papers (official harness)."""

    @abstractmethod
    def reset(self) -> None:
        """Resets internal state between episodes (history, action queue).  
        Critical for MaIL (uses history) and OFT (action chunk queue)."""

    @abstractmethod  
    def predict(self, observation: dict, task_description: str) -> PolicyOutput:  
        """A single inference call. 'observation' contains standard LIBERO keys  
        (full_image, [wrist_image], [state]). Returns a normalized PolicyOutput."""

    @abstractmethod
    def get_metadata(self) -> PolicyMetadata:  
        """Returns static metadata for provenance and results tables."""

    @abstractmethod
    def get_vram_footprint_gb(self) -> float: 
        """Peak allocated VRAM (torch.cuda.max_memory_allocated)."""
```

 
### 4.4 Module Structure

Context: The overall project structure includes `old_scripts/` (legacy environment testing, to be ignored), the documentation folder `doc` (containing this file, `SETUP_ENV.md`, `pip_lists/`). The code below details the actual testing package.

```
project_root/  
├── old_scripts/                    # legacy/sandbox scripts (ignore)  
├── doc/                            # documentation and setup rules  
│   ├── arch_design.md              # this file  
│   ├── SETUP_ENV.md                # environment setup instructions  
│   └── pip_lists/                  # pip requirements per environment  
└── testbench/                      # standalone package, imports repos as libraries  
    ├── __init__.py  
    ├── core/  
    │   ├── policy_wrapper.py       # ABC PolicyWrapper \+ dataclasses (the contract)  
    │   ├── evaluator.py            # LiberoEvaluator (Facade) \+ rollout loop (Template Method)  
    │   ├── metrics.py              # latency/throughput/percentiles/Pareto calculations  
    │   ├── benchmark.py            # isolated real-time micro-benchmark (batch=1, warm-up)  
    │   └── factory.py              # policyFactory (registry model_id -> wrapper)  
    ├── wrappers/  
    │   ├── openvla_wrapper.py      # adapter for openvla/openvla (predict_action)  
    │   ├── oft_wrapper.py          # adapter for openvla-oft (get_vla_action)  
    │   └── mail_wrapper.py         # adapter for ALRhub/MaIL  
    ├── io/  
    │   ├── results.py              # per-episode JSON serialization \+ aggregated CSV  
    │   └── provenance.py           # environment metadata collection (versions, commits, GPU)  
    ├── config/  
    │   └── suites.py               # mapping suite -> checkpoint, unnorm_key, task list  
    └── run_eval.py                 # CLI entrypoint: \--model \--suite \--quant \--num-cameras ...
```
---

## 5. Metrics (Operational Definitions)
### 5.1 Success Rate (Part A — Official Harness)

* Binary success per episode, averaged over 20 rollouts/task, totaling 200/suite.  
* Reported as mean +- std across 3 seeds (where applicable) or across the 200 episodes.  
* Secondary metric: steps-to-success (average length of successful episodes) = behavior efficiency.

### 5.2 Real-Time (Part B — Isolated Micro-benchmark AND In-the-loop)
**Two distinct measurements, both reported:**
1. **Isolated Micro-benchmark** (benchmark.py): batch=1, W=15 discarded warm-up steps, N\>=100 timed iterations using torch.cuda.synchronize() before/after. Controlled environment, no simulator overhead.  
   → *Pure model latency.*  
2. **In-the-loop** (during actual eval): Latency per call recorded episode by episode.  
   → *Latency under realistic conditions (including observation preprocessing overhead).*

**Metrics reported per (model × quant × n_cameras):**

* Latency per call: mean, median, std, **p95, p99** (jitter matters for real-time control).  
* Time-to-first-action (cold start, measured separately).  
* **Architecture-dependent throughput definition:**  
  * *Closed-loop* (OpenVLA, MaIL): throughput = 1 / latency; actions/s = feedback frequency.  
  * *Chunked* (OFT, K=8): effective actions/s = K / latency; feedback frequency = 1 / latency (these are distinct\!).  
* Peak VRAM (GB).  
* Number of parameters.

### 5.3 Final Output
Pareto frontier: x-axis = effective control frequency (or latency), y-axis = success rate, bubble size = VRAM. This will be the key figure of the thesis.
---

## 6. Output Format (Comparability + Reproducibility)
For each run (one model, one suite, one config):

```
results/  
└── {model_id}_{suite}_{quant}_{ncam}cam_{timestamp}/  
    ├── episodes.jsonl          \# One JSON line per episode (success, steps, per-step latencies)  
    ├── summary.csv             \# Aggregated row: SR, aggregate latencies, throughput, VRAM  
    └── provenance.json         \# Package versions, commit hashes, GPU info, full config, seed
```
**episodes.jsonl** (One line per episode):

```json  
{"suite":"spatial","task_id":3,"trial":12,"seed":0,"success":true,"num_steps":187,  
 "latencies_s":\[0.11,0.09,...\],"lat_mean_s":0.098,"is_chunked":true,"chunk_size":8}
```

**summary.csv** (Aggregated, one row per run — comparable across models):
```
model_id,suite,quant,n_cam,seed,success_rate,n_episodes,lat_mean_ms,lat_p95_ms,lat_p99_ms,effective_aps,feedback_hz,vram_peak_gb,num_params,mean_steps_success
```

**provenance.json**: Complete snapshot (refer to SETUP_ENV.md) — versions, commits, GPU, driver/CUDA, complete CLI config.  
A separate aggregate.py script combines the various summary.csv files (produced by launches in different venvs) into a single master table and generates the Pareto frontier.
---

## 7. Extensibility (Adding a 4th Model)

Thanks to the Strategy \+ Factory patterns, adding a model only requires:

1. Creating wrappers/new_wrapper.py that implements PolicyWrapper.  
2. Registering it in factory.py (PolicyFactory.register("new_model", NewWrapper)).  
3. Adding its suites/checkpoints in config/suites.py.

No modifications are needed for evaluator.py, metrics.py, benchmark.py, or run_eval.py (Open/Closed Principle).