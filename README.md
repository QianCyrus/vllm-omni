# VAE batch parallel decode: experiment evidence

This repository records the tests for an opt-in VAE batch decode mode in
vLLM-Omni. The mode assigns complete images to the existing worker GPUs, then
gathers the results in image order. Each image uses its native decoder and
retains its full spatial context.

**Status: component, HTTP and boundary diagnostics completed.** The completed HTTP control/batch/control run reduced latency
by 15.42–15.91% for two images and 19.76–19.91% for four images. All 52 B2/B4
images per arm pair matched exactly; the separate B1 check did not. A follow-up
found different inputs already before VAE decode, while each B1 fallback matched
direct native decode on its own input exactly. Both earlier
failed HTTP attempts are retained. Component and HTTP results have separate
measurement scopes below.

## Source, hardware and software

- vLLM-Omni base: `43e507117f04a86f2df8d405cbe03c1b1642eac9`.
- Candidate: `1ccbc20901149b803a135984c2c3e4847fe981dd`.
- Model: `black-forest-labs/FLUX.2-klein-4B`, revision
  `e7b7dc27f91deacad38e78976d1f2b499d76a294`.
- Distributed component hardware: 4 × NVIDIA A100-SXM4-40GB, one NVLink node.
  The earlier scaling probe used one active A100.
- vLLM `0.30.0`, PyTorch `2.13.0+cu130`, CUDA build `13.0`, NCCL `2.29.7`,
  Diffusers `0.40.0`, Transformers `5.14.1`.
- Installed vLLM-Omni package metadata: `0.30.0rc1+kvbench`; the served source
  is pinned to the candidate commit and file hashes above.
- BF16 VAE weights and inputs; 1024×1024 output. Tiling and slicing were off
  in the measured component comparison.

The source patches and file hashes are in [source/](source/). The component
snapshot and candidate have the same executable Python AST; the later changes
add typing annotations and comments. The final service manifest identifies the
candidate exactly. No model weights are included.

## Bottleneck and measured value

Without batch distribution, each worker decodes the full image batch. Splitting
complete images can reduce per-rank decoder work and activation memory when
several GPUs already serve the request. It adds output gathering and broadcast,
so it has no universal speedup guarantee. It does not combine independent
serving requests, reduce denoising steps, or change precision.

The same-workload component test used real pretrained VAE weights and fixed
synthetic latents on both arms. Both used the same four workers. The baseline
used native full-batch decode on each worker; batch mode split that same batch
over the four workers. Timings include the gather and broadcast.

| Images per batch | Baseline mean ± std (ms) | Batch mode mean ± std (ms) | Latency reduction |
| ---: | ---: | ---: | ---: |
| 1 | 153.659 ± 0.128 | 153.717 ± 0.159 | -0.04% |
| 2 | 380.572 ± 12.042 | 157.172 ± 3.865 | 58.70% |
| 4 | 696.512 ± 1.163 | 155.839 ± 1.459 | 77.63% |
| 5 | 857.028 ± 1.162 | 377.247 ± 0.320 | 55.98% |

Each arm has two rounds, two warmups and eight measured decodes per round, in
A-B-A-B order. Each sample is the slowest rank's wall time. Standard deviation
uses all 16 measured samples per arm. B1 takes the native fallback and is neutral
within the observed variation. No case or outlier was dropped.

| Images per batch | Baseline peak allocated (MiB) | Batch mode peak allocated (MiB) |
| ---: | ---: | ---: |
| 1 | 2612.8 | 2612.8 |
| 2 | 4277.3 | 2612.8 |
| 4 | 8375.3 | 2612.8 |
| 5 | 10424.3 | 4277.3 |

Memory is the highest per-rank PyTorch allocated peak in the VAE-only process,
not full-model memory or device-wide reserved memory. All four ranks returned
the complete ordered batch with matching dtype and shape. Max absolute error,
MAE and RMSE were zero for B1/B2/B4/B5. This does not establish exact equality
for other VAE models, precisions or execution backends.

## Included runs and checks

| Record | Scope | Result |
| --- | --- | --- |
| [9723410](runs/9723410/) | Single-GPU VAE scaling and batch-decomposition probe; real weights, synthetic latents | Passed; raw timings, metadata, numerical checks and operator summary included. The large diagnostic trace is omitted with its hash recorded in `curation.json`. |
| [9723523](runs/9723523/) | Four-rank native VAE component A/B including communication | Passed, exit 0; all raw timing/correctness JSON, logs, source checks and cleanup receipts included. |
| [9723702](runs/9723702/) | First compiled HTTP launch, control arm | **Failed before readiness**, exit 1; no valid HTTP performance result. Original harness, logs and failure/cleanup receipts are preserved. |
| [9723802](runs/9723802/) | Second compiled HTTP attempt | First control completed 19 requests, including 12 timed requests. The next arm failed at the launcher's port probe; whole job exit 1. **No complete A/B result.** Raw responses, images, timing, cleanup and original harness are preserved. |
| [9723875](runs/9723875/) | Compiled native HTTP control/batch/control | Completed, exit 0; 19 requests per arm, valid timing records and successful cleanup. B2/B4 comparisons are exact; the B1 check is nonexact and investigated separately below. Full raw responses, retained images, analysis and original harness are included. |
| [9723970](runs/9723970/) | Separate unscored native stage diagnostic | Completed, exit 0; VAE/DiT stage observations and exact agreement with scored B2/B4 images. Rank-log alignment is incomplete; analysis status is `VALIDATED_DIAGNOSTIC_WITH_LIMITATIONS`. |
| [9724061](runs/9724061/b1-diagnostic-analysis.md) | Separate unscored B1 boundary diagnostic after original request history | Completed, exit 0; inputs differ before VAE; fallback equals same-input native decode exactly in each arm. No whole-pipeline bitwise claim. |
| [CPU tests](validation/pytest-batch-cpu.log) | Focused CPU/Gloo tests | 42 passed, 15 warnings in 42.17 seconds; not a whole-repository test pass. |
| [Mypy comparison](validation/mypy-comparison-after.json) | Exact base versus final candidate | 32 unchanged base errors, zero new errors. |
| [Pre-commit](validation/precommit-final.log) | Applicable changed-file hooks | Non-mypy hooks passed. Mypy was skipped in this combined invocation after its separate comparison. |
| [Full precheck](validation/precheck-full.md) and [closure](validation/precheck-closure.md) | Source/evidence review and final evidence addendum | No high-confidence code correctness blocker; later full-model, stage and B1 boundary evidence is recorded in the addendum with its limits. |

Run 9723702 failed because the API server tried to create a speaker cache below
an unwritable default home directory. Its `scored: true` metadata records the
intended run mode, not a valid score. The portable server command below sets
`SPEAKER_SAMPLES_DIR` under `/tmp`.

Run 9723802 shut down the control server successfully, but the next launcher's
plain socket bind failed with `EADDRINUSE`. A closed server can leave TCP
connections in `TIME_WAIT`. The retry changes the probe to use `SO_REUSEADDR`,
matching the server's reuse behavior. The [Linux preflight receipt](validation/port_preflight_validation.json)
confirms that this permits rebinding after a closed server while still rejecting
a live listener. Retry 9723875 completed all three arms.

**Stage metric units:** the original HTTP client treated all scalar values in
`metrics.stage_durations` as seconds. That field contains mixed units, so the
derived `stage_0_gen_ms_ms` values have an extra factor of 1000. HTTP `wall_ms`
and the raw server metrics are unaffected. The corrected
[offline analyzer](scripts/analyze_service.py) retains both original views and
adds normalized fields with their unit provenance. It never guesses unknown
units. The [fix receipts](validation/service-metric-units.md) include synthetic
and real-response checks. No scored request was rerun or rewritten for this fix.

The public [v2 client](scripts/bench_image_service_v2.py) changes only metric
normalization. Its timer, payload, request loop and image checks match the scored
client, as verified in [the comparison receipt](validation/service_client_v2_local_validation.json).
The original client remains in every run's `harness/` and in
`scripts/bench_image_service.py`. The historical note describes preparation
before the scored job finished; v2 is now the public reproduction client.

PNG files under the run directories are model-generated test outputs, not
reference ground truth. Their model revision is pinned above; each request's
prompt, seed, shape and generation settings are retained in `requests.jsonl`.

The archived precheck is a receipt from its review time; it is not rewritten
to turn a pending or failed experiment into a pass.

## Reproduce the focused CPU tests

The 42-test result covers these three modules. Run from the candidate checkout
with its test dependencies, including `pytest-mock`, installed:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  -p pytest_mock -p no:cacheprovider -q -o addopts='' \
  tests/diffusion/distributed/test_vae_batch_parallel.py \
  tests/diffusion/distributed/test_distributed_vae_executor.py \
  tests/diffusion/distributed/test_vae_patch_parallel.py \
  -m 'core_model and cpu'
```

The recorded test environment disabled automatic plugin discovery and loaded
`pytest-mock` explicitly. The archived precheck's earlier two-module suggestion
does not describe the complete scope of the 42-test execution.

## Reproduce the component test

Use the pinned candidate checkout, its supported Python environment, four
available GPUs, and a local copy of the pinned model snapshot. In this example,
`VAE_SOURCE` is that checkout and `VAE_MODEL` is the model snapshot directory.
Run commands from this evidence repository.

```bash
export VAE_SOURCE=/path/to/vllm-omni-candidate
export VAE_MODEL=/path/to/FLUX.2-klein-4B
export PYTHONPATH="$VAE_SOURCE"
mkdir -p results/component-repro
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=4 \
  scripts/bench_vae_distributed.py \
  --model "$VAE_MODEL" --output results/component-repro
python scripts/analyze_component.py results/component-repro
```

The component script and analyzer are copied without changes. The earlier
scaling script is also retained; it measures first, then captures a separate
diagnostic trace after the scored samples.

## Portable service reproduction

These commands reproduce the compiled HTTP setup independently of PJM. The
recorded run used the original client; the public v2 client fixes only metric
reporting. Per-arm metadata contains the exact recorded server/client arguments.

Use the same candidate source, model snapshot and environment on both arms.
For the first control, set `VAE_ARM=control`, `VAE_RUN_LABEL=control-a`, and
`VAE_DEGREE=1`. For the candidate, use `batch`, `batch`, and `4`. Restart the
server for each arm, use fresh output/cache directories, and run the control
again as `control-b`. All other settings must remain equal. This comparison
changes the VAE degree on one source snapshot; a main-branch comparison is a
separate check.

### 1. Server

Set `VAE_SOURCE` and `VAE_MODEL` as above. Run this block in the server terminal
from the evidence repository. Keep its log until shutdown completes.

```bash
export VAE_EVIDENCE="$PWD"
export VAE_SOURCE_SHA=1ccbc20901149b803a135984c2c3e4847fe981dd
export VAE_ARM=control
export VAE_RUN_LABEL=control-a
export VAE_DEGREE=1
export VAE_CACHE="/tmp/vae-batch-${VAE_RUN_LABEL}"
export PYTHONPATH="$VAE_SOURCE"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export XDG_CACHE_HOME="$VAE_CACHE/xdg"
export FLASHINFER_WORKSPACE_BASE="$VAE_CACHE/flashinfer"
export TRITON_CACHE_DIR="$VAE_CACHE/triton"
export TORCHINDUCTOR_CACHE_DIR="$VAE_CACHE/inductor"
export VLLM_CACHE_ROOT="$VAE_CACHE/vllm"
export CUDA_CACHE_PATH="$VAE_CACHE/cuda"
export SPEAKER_SAMPLES_DIR="$VAE_CACHE/speakers"
mkdir -p "$XDG_CACHE_HOME" "$FLASHINFER_WORKSPACE_BASE" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$CUDA_CACHE_PATH" \
  "$SPEAKER_SAMPLES_DIR" "$VAE_EVIDENCE/results"
python -m vllm_omni.entrypoints.cli.main serve "$VAE_MODEL" \
  --omni --host 127.0.0.1 --port 8091 --served-model-name klein-vae-bench \
  --model-class-name Flux2KleinPipeline --num-gpus 4 --dtype bfloat16 \
  --tensor-parallel-size 4 --data-parallel-size 1 --pipeline-parallel-size 1 \
  --ulysses-degree 1 --ring-degree 1 --allgather-degree 1 --cfg-parallel-size 1 \
  --vae-parallel-mode batch --vae-patch-parallel-size "$VAE_DEGREE" \
  --vae-fast-path off --cache-backend none \
  --diffusion-attention-backend TORCH_SDPA --log-stats \
  --diffusion-compile-granularity regional --diffusion-compile-dynamic \
  > "$VAE_EVIDENCE/results/${VAE_RUN_LABEL}-server.log" 2>&1
```

Wait for `/health` readiness before sending requests. Compilation and warmup
are excluded from the measured requests; readiness alone does not warm the
tested shapes. For an eager comparison, change both arms together: remove the
two compile flags and add `--enforce-eager`. Report eager results separately.

### 2. Fixed request

In another terminal, after readiness:

```bash
curl --fail-with-body --max-time 300 \
  http://127.0.0.1:8091/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"klein-vae-bench","prompt":"A small red sailboat on a calm blue lake, distant green hills, soft morning light, realistic photograph.","n":2,"size":"1024x1024","num_inference_steps":4,"guidance_scale":1.0,"seed":8137,"generator_device":"cpu","response_format":"b64_json","output_format":"png","output_compression":100,"return_stage_metrics":true}' \
  -o /tmp/vae-batch-request.json
```

This is a standalone API example. Skip this extra smoke request when matching
the recorded benchmark's request order. `n=2` means one request for two images,
not concurrency two.

### 3. Benchmark client

Run from the evidence repository in the client terminal. Set the arm and run
label to match the server. The output directory must not already exist.

```bash
VAE_ARM=control
VAE_RUN_LABEL=control-a
python scripts/bench_image_service_v2.py \
  --base-url http://127.0.0.1:8091 --model klein-vae-bench \
  --output "results/${VAE_RUN_LABEL}/http" \
  --arm "$VAE_ARM" --source-sha 1ccbc20901149b803a135984c2c3e4847fe981dd \
  --batches 2 4 --warmup 2 --repeats 6 --seed 8137 --steps 4 \
  --width 1024 --height 1024 --request-timeout 600 --quality-cases 2
```

The client measures HTTP POST through the complete response read. This includes
server generation, PNG/base64 work and response transport. It excludes local
JSON parsing, image checks and file writes. It checks image count and size,
hashes every decoded image, and saves selected images for cross-arm comparison.
B1 and alternate prompt/seed requests are correctness checks outside the timed
samples. Six samples do not support a useful p99 claim.

## Harness identity and audit

The portable benchmarks and clients live in [scripts/](scripts/). The exact
cluster launcher and PJM wrappers are in [cluster_harness/](cluster_harness/).
They retain their scheduler guard and original behavior. The service wrapper
sets the speaker cache for the retry. They are historical cluster harnesses;
use the raw commands above on other GPU hosts.

The current launcher hash is
`3f605c650e1f4d4e0ab9cf68409383e8f9e4ae9c3d8b98f3a58da7c53f7bc921`.
Its source list includes the actual
`vllm_omni/diffusion/models/flux2_klein/flux2_klein_transformer.py`.
The older launchers used by failed jobs 9723702 and 9723802 remain unchanged
under each run's `harness/` directory. Do not use the newer file to verify an
older run. The latest launcher differs from 9723802's version only by the
`SO_REUSEADDR` option and its two comment lines; source and client are unchanged.

`curation.json` binds copied files to their original archive paths and hashes.
Original log paths are retained as evidence. Files were not edited to remove
failed runs or make historical hash checks appear current. Model weights,
environments, credentials and SSH configuration are excluded.

Use `sha256sum --check SHA256SUMS` (or `shasum -a 256 -c SHA256SUMS` on
macOS) to check this snapshot. The whole-job service analyzer checks the original job
layout and harness receipts; manual raw-server runs do not fabricate those
scheduler receipts.

## Final HTTP results

Run [9723875](runs/9723875/) used the same native Klein-4B source, model,
four GPUs, TP4/SP1, BF16, 1024×1024 output, four denoising steps, seed 8137 and
CPU generator on all three arms. Regional dynamic transformer compilation was
enabled; VAE fast paths and the cache backend were off. The sole server-option
change was VAE degree 1 / 4 / 1 in batch mode. Degree one uses native decode.
This is a feature toggle on the candidate source, not a separate main-checkout
comparison. Each arm starts a fresh server and cache directory.

| Images per request | Control A mean ± sample std (ms) | Batch mode (ms) | Control A2 (ms) | Latency reduction vs both controls |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1416.129 ± 8.113 | 1190.840 ± 2.633 | 1407.962 ± 1.384 | 15.42–15.91% |
| 4 | 2703.025 ± 3.082 | 2164.905 ± 7.291 | 2698.090 ± 2.775 | 19.76–19.91% |

Each table cell uses six measured requests after two warmups for its batch size.
Timing covers HTTP POST through the complete response read, including generation,
PNG/base64 and transport. Local image checking and file writes are excluded.
The control repeat changed by -0.58% at B2 and -0.18% at B4. This is sequential
single-request latency; it is not a concurrent-serving throughput or p99 result.

The [quality audit](runs/9723875/quality-review.md) verifies all corresponding
B2/B4 warmup, measured and alternate prompt/seed outputs: **52/52 decoded images
per arm pair match exactly**. The separate, unscored B1 request differs between
all three processes. Candidate/control RGB RMSE is 2.083 and 1.973 on 0–255
channels, versus 1.334 between the two controls. Thus B1 is outside that same-job
control envelope. The previous control from failed job 9723802 matches the
candidate B1 exactly, showing baseline process variation, but this does not
identify the cause or establish B1 noninferiority. The analyzer correctly reports
`COMPLETE_NONEXACT_REVIEW_REQUIRED`.

## Separate stage diagnostic

Run [9723970](runs/9723970/diagnostic-analysis.json) used the same compiled
settings in a separate control/batch run, with native profiler hooks enabled.
Each cell below contains three warmed diagnostic requests. These are synchronized
host wall observations from the native profiler, including VAE collectives;
they are not CUDA-event/kernel timings or scored service latencies.

| Images | Native VAE mean ± sample std (ms) | Batch VAE (ms) | Native DiT (ms) | Batch DiT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 375.124 ± 0.058 | 155.330 ± 0.048 | 780.768 ± 1.241 | 781.196 ± 0.419 |
| 4 | 695.384 ± 0.297 | 155.656 ± 0.237 | 1524.681 ± 5.966 | 1522.709 ± 1.938 |

The VAE decrease is about 220 ms at B2 and 540 ms at B4, while DiT time remains
close. This agrees with the separate scored HTTP decrease of 217–225 ms and
533–538 ms, supporting VAE decode as the source of the gain. Times from the two
runs are not subtracted to claim an exact breakdown of HTTP latency.

Every corresponding diagnostic image matches the scored run in decoded RGB.
The raw response stage accumulations are present and valid. Interleaved rank
logs cannot establish all four transformer calls for every request on every
rank, so the analysis reports `VALIDATED_DIAGNOSTIC_WITH_LIMITATIONS`. No missing
rank times are filled in and no maximum-rank timing is inferred.

## B1 boundary diagnostic

Separate run [9724061](runs/9724061/b1-diagnostic-analysis.md) replayed the same
16-request B2/B4 history before the B1 request, with seed 8137 and four steps.
The pre-VAE BF16 inputs differ between the fresh control and candidate
processes. Each arm's B1 fallback exactly matches direct native Diffusers decode
on its own unchanged tensor: max error, MAE and RMSE are zero. Startup dummy
captures are excluded by matching the actual HTTP request ID and order.

This locates a reproduced difference before VAE decode and supports native
fallback correctness on these inputs. It does not identify the first upstream
operation that differs, rule out indirect process-history effects, or turn the
original B1 comparison into an exact result. No B1 speedup or universal bitwise
quality claim is made. Input tensors, frozen hooks, captures and hash checks
are included; no model weights are included.

## Recheck the included evidence offline

Install Pillow, then run from this evidence checkout:

```bash
python scripts/analyze_component.py runs/9723523
python scripts/analyze_service.py runs/9723875
python scripts/analyze_diagnostic.py runs/9723970 --scored-run runs/9723875
python scripts/analyze_b1_diagnostic.py runs/9724061 --scored-run runs/9723875
```

The service and stage analyzers intentionally return exit code **2**, for
`COMPLETE_NONEXACT_REVIEW_REQUIRED` and `VALIDATED_DIAGNOSTIC_WITH_LIMITATIONS`.
The B1 analyzer returns **0**, `COMPLETE_BOUNDARY_EVIDENCE`, with localization
`DIVERGENCE_PRESENT_BEFORE_VAE`. These are different scopes. Do not reinterpret
exit 2 as a missing run or rewrite it as an all-output pass. The commands may
regenerate analysis JSON files with local absolute paths; raw artifacts are
unchanged. The original archived analysis is bound by `SHA256SUMS`.

AI assistance: Codex assisted with implementation, tests, benchmark scripts,
evidence review and this README. This repository uses the Apache 2.0 license;
the [LICENSE](LICENSE) is copied from the tested vLLM-Omni checkout.
