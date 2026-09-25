# FLUX.2-klein-4B full-model experiment

This package tests [vLLM-Omni PR #8097](https://github.com/vllm-project/vllm-omni/pull/8097) through the normal `Omni.generate` API, including text encoding, denoising, VAE decoding, and image output. Head reduced warm request latency by **2.98% and 2.30%** against the two separate baselines in the recorded A/B/A run. Saved outputs matched exactly. These results apply to the existing native four-GPU path, which has the sequence-parallel configuration issue described below; they do not establish single-GPU output equivalence.

## Setup

| Item | Value |
| --- | --- |
| Base | `ed5fc2b83b5dcfda6bae7132b4ac87a4b2ca54c2` |
| Head | `99c27812387ab78ed26d354bc10df8595fb173d8` |
| Model | `black-forest-labs/FLUX.2-klein-4B` |
| Model revision | `e7b7dc27f91deacad38e78976d1f2b499d76a294` |
| Hardware | Four active NVIDIA A100-SXM4-40GB GPUs, one node, NVLink |
| Allocation | One full eight-GPU node reserved and billed; the experiment uses four GPUs |
| Workload | One prompt, two output images, 1024×1024, four denoising steps, seed 8137 |
| Execution | BF16, eager, `TORCH_SDPA`, AllGather degree 4; other parallel degrees 1 |
| Disabled | Quantization, offload, model cache, prompt embedding cache, CUDA Graphs |
| Scored order | Base A → head → base B; two warmup and six timed requests per job |

All three scored runs record Python 3.12.13, vLLM 0.30.0, PyTorch 2.13.0+cu130, CUDA build 13.0, NCCL 2.29.7, diffusers 0.40.0, and transformers 5.14.1. The installed vLLM-Omni label is `0.30.0rc1+kvbench`; the source SHAs above identify the code tested. The GPU records show driver 535.54.03; the launcher uses CUDA 13.3 compatibility libraries.

`TORCH_SDPA` selects the framework backend. This experiment does not explicitly force one PyTorch SDPA kernel.

## Measurement and status

Scored latency is wall time around blocking `Omni.generate`, including text encoding, denoising, VAE decoding, and output transport. It excludes model loading and PNG writing. Each request returns two images; this is not two concurrent requests. Every request uses a fresh CPU generator with the same seed.

Both baseline jobs are kept separate. The table reports mean ± sample standard deviation across six timed requests per job. These are within-job samples, not six independent job repetitions. [Full statistics and checks](analysis/summary.md) include medians, ranges, configuration comparisons, and output comparisons.

| Run | Kind | Status | Mean ± std request latency, s | Head reduction vs this baseline |
| --- | --- | --- | --- | --- |
| Base A, [9718337](runs/9718337/) | Scored, 2 warmup + 6 timed | PASS | 1.128337 ± 0.015133 | 2.98% |
| Head, [9718379](runs/9718379/) | Scored, 2 warmup + 6 timed | PASS | 1.094734 ± 0.004220 | — |
| Base B, [9718423](runs/9718423/) | Scored, 2 warmup + 6 timed | PASS | 1.120519 ± 0.005266 | 2.30% |

All scored runs completed eight requests, passed source checks, completed cleanup, and exited with code 0. Model, workload, dependency versions, hardware, backend, and execution configuration match. The prepared source trees differ in exactly two files: the AllGather helper and its targeted test. Only the helper differs among the nine runtime files fingerprinted by the driver.

Each baseline-to-head comparison covers 12 measured image pairs. PNG file hashes match the recorded hashes, and remote decoded-pixel comparisons report zero differences. Independent local checks also found identical PNG bytes at each output position across every saved request, including warmups and the head diagnostic. This concerns the same prompt and seed, not general image quality. The worker-reported peak reserved metric is **23,346 MiB** in every recorded request; no full-model memory reduction is observed by this metric.

## Native path and separate diagnostics

Head diagnostic job [9718248](runs/9718248/) passed with one warmup and one measured request. Its latency is not a scored result. Hooks observed the native AllGather-KV path on all four ranks: input Q/K/V `[2, 1536, 24, 128]`, gathered K/V `[2, 6144, 24, 128]`, and no separate joint tensors. Each rank executes 25 target preparation calls per transformer forward, or 100 during the four-step measured request.

Source review found an existing configuration omission in both arms: [`Flux2KleinPipeline` creates the transformer without `od_config`](https://github.com/vllm-project/vllm-omni/blob/ed5fc2b83b5dcfda6bae7132b4ac87a4b2ca54c2/vllm_omni/diffusion/models/flux2_klein/pipeline_flux2_klein.py#L258), so [the transformer defaults to sequence-parallel size 1](https://github.com/vllm-project/vllm-omni/blob/ed5fc2b83b5dcfda6bae7132b4ac87a4b2ca54c2/vllm_omni/diffusion/models/flux2_klein/flux2_klein_transformer.py#L861). The registered sharding hooks and global attention groups still use degree 4. Consequently, replicated text is concatenated with the image shard before gathering: 1,024 image tokens plus 512 text tokens per rank become 6,144 gathered tokens. Both native arms share this behavior. These runs do not qualify the separate joint-text path or output equivalence to sequence-parallel size 1. Fixing that omission and testing the corrected model path are separate work.

Diagnostic runs enable pipeline profiling and Python hooks. Hook synchronization and file writes affect timing; keep their stage measurements separate from scored latency. Transformer timings are nested inside pipeline forward and must not be added to its total. Use the [phase-separated head report](derived/9718248/diagnostic_phases.md) for attribution: the general analyzer's all-forward average includes startup and warmup. Baseline diagnostic [9718466](runs/9718466/) also passed with one warmup and one measured request. The [stage comparison](derived/stage_comparison.md) excludes startup and warmup:

| Measured stage, rank 0 | Base (ms) | Head (ms) |
| --- | ---: | ---: |
| Transformer, four steps | 622.759 | 594.786 |
| Text encoder | 42.667 | 40.403 |
| VAE decode | 375.345 | 375.870 |
| Enclosing pipeline | 1058.963 | 1028.383 |

The roughly 28 ms transformer difference is consistent with the scored request savings. Each arm has only one instrumented diagnostic request; this is not another scored speedup estimate. The components are nested inside the pipeline total. Both diagnostics completed normally and exited with code 0; head workers needed the executor's terminate fallback after its graceful wait. The two sources have identical teardown code, and neither diagnostic logged kill or incomplete-cleanup warnings.

## Reproduce

Use clean source checkouts at the two SHAs above, the pinned local model, and a compatible environment. Run from this directory on allocated GPUs. Adapt the three paths below to the local setup. Each iteration starts a fresh process and loads its own model.

```bash
KV_MODEL_SOURCE_ROOT=/path/to/checkouts
KV_MODEL_MODEL_DIR=/path/to/FLUX.2-klein-4B
KV_MODEL_PYTHON=/path/to/environment/bin/python
unset KV_MODEL_DIAGNOSTIC_DIR

for run in base_a head base_b; do
  case "$run" in
    head) arm=head; sha=99c27812387ab78ed26d354bc10df8595fb173d8 ;;
    *) arm=base; sha=ed5fc2b83b5dcfda6bae7132b4ac87a4b2ca54c2 ;;
  esac
  PYTHONPATH="$KV_MODEL_SOURCE_ROOT/source_$arm" "$KV_MODEL_PYTHON" bench_model.py \
    --source-root "$KV_MODEL_SOURCE_ROOT/source_$arm" --source-sha "$sha" \
    --model "$KV_MODEL_MODEL_DIR" --output "runs/$run" --world-size 4 \
    --height 1024 --width 1024 --batch 2 --steps 4 --warmup 2 --repeats 6
done

"$KV_MODEL_PYTHON" analyze_model.py --runs runs --outputdir analysis \
  --base-sha ed5fc2b83b5dcfda6bae7132b4ac87a4b2ca54c2 \
  --head-sha 99c27812387ab78ed26d354bc10df8595fb173d8 --pixel-metrics
```

Pass both SHA arguments to the analyzer: its preserved defaults refer to the earlier September 24 source pair. The Wisteria launcher `run_model4_node.pjm` retains the exact cluster paths and defaults to a short diagnostic. Scored submissions must set `KV_MODEL_MODE=scored`, `KV_MODEL_WARMUP=2`, and `KV_MODEL_REPEATS=6`, plus the intended arm. Do not enable diagnostic hooks for scored runs.

## Files and limits

- `bench_model.py`, `analyze_model.py`, `run_model4_node.pjm`, and `diagnostic_hooks/sitecustomize.py` are unchanged copies of the experiment tools.
- `experiment.json` records the source pair, model, workload, and allocation plan.
- `source_base.sha256` and `source_head.sha256` preserve the prepared source-tree manifests. Their paths are relative to the experiment root; source trees are not bundled here.
- `setup-evidence/` contains the September 24 environment package list, installation/import logs, model metadata, download checks, and model file hashes. The environment and model are reused; old source-import paths in those logs do not identify the September 25 runtime source.
- `runs/` contains all five completed raw runs. `analysis/` contains the scored comparisons and diagnostic route summaries; `derived/` contains both phase-separated diagnostic reports and their comparison. `verification_scored_aba.json` records the independent scored-run audit; `verification_all_runs.json` verifies all published run artifacts.

One prompt and one seeded batch cannot establish general image quality. Any pixel comparison applies only to the saved outputs. The corrected sequence-parallel path, single-GPU equivalence, other attention backends, and broader shapes require separate validation. Worker peak reserved memory is not the earlier operator benchmark's allocated-memory increment or a sum across four GPUs.
