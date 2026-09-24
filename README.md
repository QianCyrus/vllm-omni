# vLLM-Omni KV AllGather experiment

Benchmark scripts and full results for a proposed change from two K/V AllGather calls to one packed call in diffusion attention.

**September 24 update:** [Five-arm ablation and profiler evidence](review_20260924/README.md) now compare the exact PR base and head. The new results confirm the measured batch-2 benefit and show that layout-only is a strong alternative: packing does not consistently improve on it. The original September 23 artifacts below are unchanged.

- Hardware: 4 and 8 NVIDIA A100-SXM4-40GB GPUs, one node, NVLink.
- PyTorch 2.13.0+cu130, CUDA build 13.0, NCCL 2.29.7, Python 3.12.
- Baseline source: cb5f508212befb16b48e5760f3b7508c457d6413.
- Scope: source-extracted preparation plus PyTorch Flash SDPA, with real NCCL. No full engine or model was run.

See [reproduction instructions](benchmark/README.md), [the complete final results](benchmark/final_results.csv), and [the downloadable ZIP](kv-allgather-a100-benchmark.zip). All tested shapes are retained, including regressions. The GPU runs were performed on 2026-09-23; packaging and publishing the evidence did not rerun them.

The code PR contains only the production helper and its targeted tests. This separate branch keeps the larger benchmark evidence out of that code diff. It does not claim that every shape or backend improves.

AI assistance: Codex helped with the implementation, tests, benchmark runs, analysis, and writing.
