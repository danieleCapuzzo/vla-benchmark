"""
OpenVLA-OFT + LIBERO checkpoint test, with configurable quantization.

Uses the OFFICIAL OFT inference pipeline (get_vla, get_vla_action, GenerateConfig)
-- NOT the raw AutoModelForVision2Seq.predict_action() API, which does not handle
LIBERO-specific action normalization correctly for OFT checkpoints.

Must be run from inside the openvla-oft/ repo root.

Usage:
    cd ~/Desktop/thesis-vla/vla-benchmark/openvla-oft
    python ../test_oft_libero.py --quant none --suite spatial
    python ../test_oft_libero.py --quant 8bit --suite spatial
    python ../test_oft_libero.py --quant 4bit --suite spatial
"""

import os
import sys
import time
import argparse
import traceback

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PASS, FAIL, INFO = "  [PASS]", "  [FAIL]", "  [INFO]"


def section(title):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def ok(msg):
    print(f"{PASS} {msg}")


def fail(msg, e=None):
    print(f"{FAIL} {msg}")
    if e:
        traceback.print_exc()


def info(msg):
    print(f"{INFO} {msg}")


SUITE_MAP = {
    "spatial": (
        "moojink/openvla-7b-oft-finetuned-libero-spatial",
        "libero_spatial_no_noops",
    ),
    "object": (
        "moojink/openvla-7b-oft-finetuned-libero-object",
        "libero_object_no_noops",
    ),
    "goal": ("moojink/openvla-7b-oft-finetuned-libero-goal", "libero_goal_no_noops"),
    "long": ("moojink/openvla-7b-oft-finetuned-libero-10", "libero_10_no_noops"),
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--quant",
    choices=["none", "8bit", "4bit"],
    default="none",
    help="Quantization mode (default: none / bf16)",
)
parser.add_argument(
    "--suite",
    choices=list(SUITE_MAP.keys()),
    default="spatial",
    help="Which LIBERO suite checkpoint to test",
)
parser.add_argument(
    "--benchmark",
    action="store_true",
    help="Also run the latency/throughput micro-benchmark (Part B protocol)",
)
parser.add_argument("--n-warmup", type=int, default=5)
parser.add_argument("--n-iters", type=int, default=20)
args = parser.parse_args()

results = {}

# -- 1. Environment -----------------------------------------------------------
section("1. Environment")
try:
    import torch

    ok(f"torch: {torch.__version__} | CUDA cap: {torch.cuda.get_device_capability(0)}")
except Exception as e:
    fail("torch import failed", e)
    sys.exit(1)

# -- 2. Import OFT modules (requires LIBERO installed) -------------------------
section("2. OFT + LIBERO module imports")
try:
    from experiments.robot.libero.run_libero_eval import GenerateConfig
    from experiments.robot.openvla_utils import (
        get_action_head,
        get_processor,
        get_proprio_projector,
        get_vla,
        get_vla_action,
    )
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

    ok("OFT modules imported correctly")
    ok(f"NUM_ACTIONS_CHUNK (K): {NUM_ACTIONS_CHUNK}")
    ok(f"PROPRIO_DIM: {PROPRIO_DIM}")
except Exception as e:
    fail("OFT module import failed -- check LIBERO is installed in this venv", e)
    sys.exit(1)

# -- 3. Build config with selected quantization --------------------------------
section(f"3. Config -- suite={args.suite}, quant={args.quant}")
checkpoint, unnorm_key = SUITE_MAP[args.suite]
info(f"Checkpoint: {checkpoint}")
info(f"Unnorm key: {unnorm_key}")

load_in_8bit = args.quant == "8bit"
load_in_4bit = args.quant == "4bit"

try:
    cfg = GenerateConfig(
        pretrained_checkpoint=checkpoint,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key=unnorm_key,
    )
    ok("GenerateConfig built successfully")
except Exception as e:
    fail("GenerateConfig construction failed", e)
    sys.exit(1)

# -- 4. Load checkpoint ----------------------------------------------------------
section("4. Checkpoint loading")
try:
    t0 = time.perf_counter()
    vla = get_vla(cfg)
    load_time = time.perf_counter() - t0
    ok(f"VLA loaded in {load_time:.1f}s")

    processor = get_processor(cfg)
    ok("Processor loaded")

    action_head = (
        get_action_head(cfg, llm_dim=vla.llm_dim) if cfg.use_l1_regression else None
    )
    ok(
        "Action head loaded"
        if action_head is not None
        else "No action head (unexpected)"
    )

    proprio_projector = (
        get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)
        if cfg.use_proprio
        else None
    )
    ok(
        "Proprio projector loaded"
        if proprio_projector is not None
        else "No proprio projector (unexpected)"
    )

    vram_after_load = torch.cuda.memory_allocated() / 1e9
    vram_peak_load = torch.cuda.max_memory_allocated() / 1e9
    ok(f"VRAM after load: {vram_after_load:.2f} GB (peak: {vram_peak_load:.2f} GB)")

    if vram_peak_load > 14.5 and args.quant == "none":
        info(
            f"WARNING: VRAM close to 16GB limit ({vram_peak_load:.2f} GB) -- "
            f"consider --quant 8bit or --quant 4bit"
        )

except Exception as e:
    fail("Checkpoint loading failed", e)
    if "out of memory" in str(e).lower():
        info("OOM -- try --quant 8bit or --quant 4bit, or free desktop VRAM overhead")
    sys.exit(1)

# -- 5. Single inference sanity check ---------------------------------------------
section("5. Single inference (sanity check)")
try:
    import numpy as np

    dummy_obs = {
        "full_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "state": np.random.randn(PROPRIO_DIM).astype(np.float32),
    }
    dummy_task_description = "pick up the object"

    action_chunk = get_vla_action(
        cfg,
        vla,
        processor,
        dummy_obs,
        dummy_task_description,
        action_head=action_head,
        proprio_projector=proprio_projector,
    )
    action_chunk = np.array(action_chunk)
    ok(
        f"Action chunk generated -- shape: {action_chunk.shape} "
        f"(expected: [{NUM_ACTIONS_CHUNK}, action_dim])"
    )
    info(f"First action in chunk: {np.round(action_chunk[0], 4)}")
    results["single_inference"] = True

except Exception as e:
    fail("Single inference failed", e)
    results["single_inference"] = False
    sys.exit(1)

# -- 6. Micro-benchmark (Part B protocol, only with --benchmark) -----------------
section(
    "6. Latency/throughput micro-benchmark"
    + ("" if args.benchmark else " [SKIPPED -- use --benchmark]")
)
if not args.benchmark:
    info("Use --benchmark to run this section")
else:
    try:
        info(f"Warm-up: {args.n_warmup} iterations (discarded)")
        for _ in range(args.n_warmup):
            _ = get_vla_action(
                cfg,
                vla,
                processor,
                dummy_obs,
                dummy_task_description,
                action_head=action_head,
                proprio_projector=proprio_projector,
            )
        torch.cuda.synchronize()

        info(f"Timed run: {args.n_iters} iterations")
        torch.cuda.reset_peak_memory_stats()
        latencies = []
        action_chunk = None
        for _ in range(args.n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            action_chunk = get_vla_action(
                cfg,
                vla,
                processor,
                dummy_obs,
                dummy_task_description,
                action_head=action_head,
                proprio_projector=proprio_projector,
            )
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)

        latencies = np.array(latencies)
        K = np.array(action_chunk).shape[0]

        lat_mean_ms = latencies.mean() * 1e3
        lat_p95_ms = np.percentile(latencies, 95) * 1e3
        lat_p99_ms = np.percentile(latencies, 99) * 1e3
        vram_peak_inference = torch.cuda.max_memory_allocated() / 1e9

        ok(f"Latency mean       : {lat_mean_ms:.2f} ms")
        ok(f"Latency p95        : {lat_p95_ms:.2f} ms")
        ok(f"Latency p99        : {lat_p99_ms:.2f} ms")
        ok(f"Actions per call(K): {K}")
        ok(f"Effective actions/s: {K / latencies.mean():.2f}  (K/mean_latency)")
        ok(f"Feedback frequency : {1.0 / latencies.mean():.2f} Hz  (1/mean_latency)")
        ok(f"Peak VRAM inference: {vram_peak_inference:.2f} GB")
        ok(f"Quantization       : {args.quant}")

        print("\n  Data for thesis table (Part B protocol):")
        print(f"    suite: {args.suite}")
        print(f"    quant: {args.quant}")
        print(f"    lat_mean_ms: {lat_mean_ms:.2f}")
        print(f"    lat_p95_ms: {lat_p95_ms:.2f}")
        print(f"    lat_p99_ms: {lat_p99_ms:.2f}")
        print(f"    actions_per_call: {K}")
        print(f"    effective_actions_per_sec: {K / latencies.mean():.2f}")
        print(f"    feedback_hz: {1.0 / latencies.mean():.2f}")
        print(f"    vram_peak_gb: {vram_peak_inference:.2f}")

    except Exception as e:
        fail("Benchmark failed", e)

# -- Summary ------------------------------------------------------------------------
section("SUMMARY")
print(f"  Suite tested       : {args.suite}")
print(f"  Quantization       : {args.quant}")
print(f"  Single inference   : {'PASS' if results.get('single_inference') else 'FAIL'}")
print(f"  VRAM peak (load)   : {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print()
print(
    "  Environment ready for full LIBERO eval."
    if results.get("single_inference")
    else "  Something failed -- review output above."
)
print()
