# Optional B1 boundary diagnostic: prepared, not executed

Existing evidence already supports an exact-equality claim scoped to B2/B4 and
the two alternate B2 cases. It does not support a universal B1 fallback quality
claim. `9723875/quality-review.md` preserves that distinction. This independent
probe may locate the first observed divergence at the VAE boundary; it does not
replace the quality gate or the ongoing B2/B4 stage diagnostic.

## New hook, isolated from the running diagnostic

`vae_b1_diagnostic_hooks/sitecustomize.py` is opt-in with
`OMNI_VAE_B1_DIAGNOSTIC=1`; it does not import Torch/models at interpreter startup.
It intercepts only the actual Klein pipeline and distributed Flux2 VAE modules.
The pipeline wrapper supplies request ID, seed, output count, step count and
CPU-generator metadata through a context variable. The VAE wrapper changes no
returned output and captures only batch-one calls on rank 0.

For each captured call it:

1. Copies the input latent to CPU and records dtype, shape, contiguous byte
   SHA256, original device and strides. By default it saves the input tensor as
   `.pt` and hashes that artifact (`OMNI_VAE_B1_SAVE_TENSORS=0` disables saving).
2. Calls the existing `_batch_parallel_decode` fallback exactly once, records
   its raw floating output hash, and checks that the input bytes did not change.
3. Makes **one extra** direct `AutoencoderKLFlux2.decode(self, z, ...)` call using
   the same unchanged input and arguments. It records that output's hash and
   fallback/direct max absolute difference, MAE, RMSE, finiteness and exactness.
   These values describe native floating VAE outputs before RGB postprocessing.
4. Writes one JSON record plus `rank0-b1-captures.jsonl`, with per-request hook
   hash and input-tensor file receipts. Probe failures are recorded and re-raised.
   Hook-install markers record module file hashes. B2+ and nonzero ranks take
   their original path without the extra native decode.

This adds CPU transfers and an extra decode and changes allocation/execution
history. Its latency, stage duration and peak-memory values must not be used as
performance scores. No candidate output is replaced by the recomputed output.

## Concrete two-process plan inside a separately allocated PJM job

Reuse the same source/model/TP4/SP1/BF16/TORCH_SDPA/regional dynamic compile
configuration and CUDA/module/storage setup as the scored service run. The
launcher still requires an allocated PJM job. The commands below are a prepared
plan, not a submission; the parent owns the overall allocation/time budget.

Set `TASK_ROOT`, `TASK_PYTHON`, `TASK_MODEL`, `TASK_OUT`, `TASK_CACHE` and the
remaining per-arm timeout `TASK_REMAIN` in that job. Use fresh output directories
and preserve these exact source files in the job's harness archive:
`launch_model_service.py`, `bench_image_service.py`, the enclosing PJM file,
`vae_b1_diagnostic_hooks/sitecustomize.py`, and `service_source_manifest.json`.
Also hash the frozen pipeline, distributed VAE mixin/native wrapper and installed
Diffusers native Flux2 VAE source. Record source and harness verification before
and after the two arms.

```bash
export SPEAKER_SAMPLES_DIR="$TASK_CACHE/speakers"
mkdir -p "$SPEAKER_SAMPLES_DIR"
OMNI_VAE_B1_DIAGNOSTIC=1 \
OMNI_VAE_B1_DIAGNOSTIC_DIR="$TASK_OUT/control-a/vae_boundary" \
OMNI_VAE_B1_SAVE_TENSORS=1 \
"$TASK_PYTHON" "$TASK_ROOT/launch_model_service.py" \
  --source-root "$TASK_ROOT/source_head" \
  --source-sha 1ccbc20901149b803a135984c2c3e4847fe981dd \
  --source-manifest "$TASK_ROOT/service_source_manifest.json" \
  --model "$TASK_MODEL" --output "$TASK_OUT/control-a" \
  --cache-root "$TASK_CACHE/control-a" --arm control --port 8121 \
  --batches 2 4 --warmup 2 --repeats 6 --compile-mode regional \
  --quality-cases 0 --startup-timeout 600 \
  --request-timeout 600 --run-timeout "$TASK_REMAIN" \
  --diagnostic --diagnostic-hook-dir "$TASK_ROOT/vae_b1_diagnostic_hooks"
```

Repeat once in a fresh process with `--arm batch`, output/cache/probe directories
changed to `batch`, port `8122`, and a recalculated remaining timeout. Do not
pre-create `control-a`/`batch` directories: the launcher creates them; the hook
creates their `vae_boundary` subdirectory during execution. Both arms replay the original shape history: B2 warmup 2 + repeats 6, B4
warmup 2 + repeats 6, then one B1 correctness request; no alternate cases.
All 17 requests are unscored. Keep startup captures explicitly separate.
`analyze_b1_diagnostic.py` maps all 17 HTTP rows to the ordered unique
`RequestE2EStats` IDs, then requires the seventeenth ID in exactly one capture
with seed 8137, steps 4, one output and a CPU generator.

## Interpretation and limits

Require complete arm validation, correct hook/source receipts, one attributable
HTTP B1 capture per arm after the original B2/B4 history, matching payload/shape/dtype and intact saved tensors.
Compare ordered captures across and within processes:

- Different input-latent hashes prove a difference already exists before VAE
  decode for those requests. They do not identify the earlier layer or rule out
  an additional decoder difference.
- Equal input values but different raw VAE outputs locate a difference in VAE
  execution/state for those calls. Compare fallback against the direct native
  recomputation to separate a repeatability issue from wrapper handling.
- Equal input and raw output hashes with different decoded RGB PNGs point after
  the captured VAE boundary.
- Exact fallback/direct output agreement proves that scoped same-input check;
  it does not prove every possible input or process history is equivalent.

The initial B1-only suggestion was superseded: `run_b1_diagnostic.pjm` now
replays the original 16 B2/B4 requests before B1. Even with matching history,
if it fails to reproduce the difference, retain `B1 review required` for the
original result rather than claiming the old variation is explained. No new tolerance should be selected after viewing the
probe. The CPU fixture validates hook routing, request context, exact/nonexact
comparison, skip behavior, tensor receipts and failure preservation; it is not
real-model/GPU evidence (`b1_diagnostic_hook_local_validation.json`).

## Offline command

```bash
uv run --offline --no-project --with pillow python analyze_b1_diagnostic.py \
  B1_DIAGNOSTIC_RUN --scored-run 9723875
```

The diagnostic run's `harness` directory must preserve every path listed in
`harness_before.sha256`; the copied manifest is read from the run root. Exit 0
means complete boundary evidence was audited, not that B1 is universally exact
or that native replay passed an acceptance threshold. Read `localization`,
`native_replay_exact_each_arm` and historical RGB matches. Exit 1 rejects
incomplete/invalid evidence. Startup captures are listed under
`excluded_captures` and are not used for causal comparison.
