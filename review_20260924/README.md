# Supplemental KV AllGather evidence, 2026-09-24

This experiment answers the performance questions on [vLLM-Omni PR #8097](https://github.com/vllm-project/vllm-omni/pull/8097). It compares five implementations within each job, using the same inputs, process group, dtype, and attention backend.

- Base: `3d43571b94f1683412023c45bd886f9ce30f76bd`.
- Head: `a2cae127c2933f0e754d38d3615da58660c952f1`.
- Hardware: 4 or 8 NVIDIA A100-SXM4-40GB GPUs on one node per job, with NVLink.
- Software: Python 3.12.13, PyTorch 2.13.0+cu130, CUDA build 13.0, NCCL 2.29.7. Driver 535.54.03 with CUDA 13.3 compatibility libraries.
- Model: none; synthetic MHA tensors. BF16, 32 Q/K/V heads, head dimension 128, non-causal attention, no joint tokens or mask. PyTorch Flash SDPA is forced.
- Cases per GPU count: B2/S4096 contiguous, B2/S4096 projection views, B2/S16384 contiguous, and B1/S4096 contiguous. S is the global sequence length; each rank holds S/world_size query tokens.

The script loads the unchanged base and head class/method definitions from the pinned source snapshots. It uses real NCCL collectives and CUDA attention kernels. It does not start an engine or load model weights.

## Five arms

| Arm | Implementation |
| --- | --- |
| `baseline` | Base source: gather K and V separately, batch-first |
| `layout_only` | Two gathers; make each local tensor sequence-first, then return transposed output views |
| `packing_only` | Pack K/V together but retain the batch-first gather layout |
| `head` | Head source: sequence-first packed K/V, one gather |
| `head_contiguous` | Head, followed by contiguous K/V materialization |

These are implementation comparisons, not independent terms that can be added together. For example, `layout_only` versus `head` changes both packing and output strides, so it does not measure only the cost of one collective call.

## Measurement and validation

Each arm has six rounds, with 10 warmup and 30 measured calls per sample. Arm order rotates and reverses across rounds. Each sample is the mean CUDA-event time per call, then the maximum across ranks. Standard deviation and range describe the six round means, not request latency percentiles. No timing samples were discarded.

The script measures eager preparation, eager preparation plus one local attention call, and CUDA Graph replay of preparation plus attention. Graphs use separate pools and all five remain alive during interleaved measurement. Eager preparation peak memory is `max_memory_allocated` minus entry allocation, including returned tensors. It is not total model memory or graph pool size.

Both jobs exited with code 0. All five arms in all eight GPU/configuration cases passed exact Q/K/V comparison. The maximum recorded absolute error for attention and changed-input graph replay was 0. Source hashes were unchanged and graph/process-group cleanup completed. The CPU preflight also passed 120 arm/rank/layout checks.

Read the [complete tables](analysis/summary.md), [machine-readable summary](analysis/summary.json), and [CSV](analysis/summary.csv). Individual samples, gather diagnostics, source hashes, and validation are under [runs/](runs/).

## Findings

For B2/S4096 contiguous inputs, graph preparation plus attention decreased from 1.643452 to 1.410293 ms on four GPUs (14.2%), and from 1.406003 to 1.172184 ms on eight GPUs (16.6%). Sample standard deviations were 0.001422/0.001522 ms and 0.001050/0.000349 ms, respectively.

Layout accounts for most of the measured batch-2 benefit. On four GPUs, layout-only reached 1.425692 ms, while packing-only reached 1.623592 ms. On eight GPUs, layout-only reached 1.146766 ms, faster than the current head at 1.172184 ms. At B2/S16384 on four GPUs, layout-only was also faster: 12.928762 versus 13.026083 ms. Packing is not consistently better than layout-only.

At B2/S16384, preparation peak memory decreased from 768 to 640 MiB on four GPUs and from 768 to 576 MiB on eight GPUs. Layout-only used still less: 576 and 544 MiB. At B1/S4096, head instead increased peak memory from 64 to 80 MiB on four GPUs and from 64 to 72 MiB on eight GPUs.

The four-GPU rank-0 [profiler traces](runs/9707225/profile/) cover three preparation calls per arm, collected separately from timing. They show six actual NCCL kernels for baseline/layout-only and three for each packed arm. Baseline has six global clone/copy pairs with shape `[2,4,1024,32,128]` (64 MiB each). Layout-only has six local clone/copy pairs with shape `[1024,2,32,128]` (16 MiB each). Head has no `aten::clone`/`aten::copy_` events, but still has three stack/cat packing operations. It does not eliminate all copying or reduce the K+V payload bytes.

The analyzer counts actual `kernel` trace events, excluding CUDA annotations such as `nccl:_all_gather_base`. The raw profiler summary also contains annotations; its `cuda_kernel_event_count` field counts all CUDA events and should not be interpreted as a kernel-only count. Profiler durations are not used as benchmark latencies, and nested durations are not added together.

Preparation-only eager measurements have larger spread in some cases; all samples remain available. The graph results above are more stable. Results do not establish full-model latency, throughput, image/video quality, or compatibility with every native attention backend. Materializing contiguous outputs can remove the benefit. These findings support a layout improvement, but do not establish that the current packed path is best for every shape.

## Reproduce

From this directory, in a compatible PyTorch/CUDA environment:

```bash
python bench_ablation.py --base-root source_base --head-root source_head --cpu-check

python -m torch.distributed.run --standalone --nproc-per-node=4 \
  bench_ablation.py --base-root source_base --head-root source_head \
  --output results/4gpu --iterations 30 --repeats 6 --warmup 10 --profile

python -m torch.distributed.run --standalone --nproc-per-node=8 \
  bench_ablation.py --base-root source_base --head-root source_head \
  --output results/8gpu --iterations 30 --repeats 6 --warmup 10

python analyze_ablation.py --runs results/4gpu results/8gpu --outputdir analysis-new
sha256sum -c SHA256SUMS
```

The recorded jobs used `OMP_NUM_THREADS=1`. These commands use portable paths; site-specific scheduler files are not included. `bench_ablation.py` and the source snapshots match the recorded hashes. All downloaded run artifacts were verified against their original checksums. In public trace copies, only `host_name` and `traceName` were normalized; trace events are unchanged. [provenance.json](provenance.json) records original and public trace hashes. `SHA256SUMS` covers the published files. The original September 23 evidence remains unchanged in `../benchmark/` and the original ZIP.

AI assistance: Codex prepared and reviewed the benchmark, ran the experiments and checks, analyzed the results, and drafted this report.
