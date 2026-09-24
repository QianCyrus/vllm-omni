# A100 KV AllGather benchmark

This archive contains the recorded operator microbenchmark, not a full-model benchmark. No model weights or private cluster configuration are included.

Baseline: vLLM-Omni cb5f508212befb16b48e5760f3b7508c457d6413, described by Git as v0.30.0rc1-12-gcb5f50821. The candidate is in source_final and candidate_final.patch. The baseline arm overrides the candidate helper with the original two-AllGather implementation.

Hardware: one node with 4 or 8 NVIDIA A100-SXM4-40GB GPUs and NVLink. Software: Python 3.12, PyTorch 2.13.0+cu130, CUDA build 13.0, NCCL 2.29.7. The measured cluster used driver 535.54.03 with CUDA 13.3 forward-compatibility libraries. Use a compatible CUDA driver setup on the reproduction host. The vLLM runtime is not imported. The Attention backend is forced to PyTorch Flash SDPA.

Run from the directory containing benchmark/ after extraction, in a matching environment with PyTorch and pytest:

```bash
KV_SOURCE_ROOT=benchmark/source_final python -m pytest -q benchmark/cpu_tests.py

python -m torch.distributed.run --standalone --nproc-per-node=4 \
  benchmark/bench_final.py --source-root benchmark/source_final \
  --output results/4gpu --iterations 20 --repeats 4 --warmup 5 --graphs

python -m torch.distributed.run --standalone --nproc-per-node=8 \
  benchmark/bench_final.py --source-root benchmark/source_final \
  --output results/8gpu --iterations 20 --repeats 4 --warmup 5 --graphs
```

On a shared cluster, run GPU commands inside a scheduler allocation. These paths are a portable form of the recorded commands. Packaging the evidence did not rerun the GPU benchmarks.

The AST loader executes selected source definitions to avoid serving dependencies; it uses real NCCL collectives and CUDA Attention. bench.py is preserved because the recorded CPU loader imports its load_source function; the final GPU benchmark entry point is bench_final.py.

Each GPU count has 26 cases. Every case uses 32 Q heads, head dimension 128, BF16, and non-causal Attention. Cases cover batch 1/2, global sequence length 1024/4096/16384, KV heads 4/32, contiguous inputs/projection views, and front/rear joint tokens. Joint cases use 16 joint queries and 24 joint K/V tokens.

Each arm has four measured rounds, five warmup iterations per round, and 20 measured iterations per round. Graph capture has three additional warmup calls. Each timing sample takes the maximum across ranks; reported values use the median across rounds. Eager arm order rotates; graph arms have separate captures. Changed-input replay is checked before timing.

results/ contains the original per-case JSONL data and validation files. The metadata files retain the original values except source/output paths, which were replaced with portable paths. final_results.csv includes only the two final runs. Memory increments from eager preparation and CUDA Graph capture are separate metrics. Negative latency reductions are valid regressions; do not omit them.

cpu_tests_final_python312.log records 19 passed in 49.60s. These are extracted targeted tests, not full-engine pytest. The source snapshot and scripts are unchanged copies of the experiment files. SHA256SUMS covers every archive file except itself.
