# Offline correction of frozen client metric units

The completed control arm of job `9723802` exposed a client-only normalization
bug. The raw HTTP response and measured HTTP wall times are unaffected. The
frozen client, model source, and original request artifacts remain unchanged.

`bench_image_service.py:time_metrics_ms()` treats every scalar inside
`stage_durations` as seconds. Current frozen source mixes units there:
`omni_base.py` merges orchestrator timing fields and `stage_N_gen_ms` as
milliseconds, while `diffusion_pipeline_profiler.py` records named native
pipeline stages using `perf_counter()` seconds.

| Raw field | Raw unit | Offline action |
| --- | --- | --- |
| Any scalar ending `_ms` | milliseconds | Preserve value; append no suffix |
| Any scalar ending `_s` | seconds | Multiply once by 1000; replace suffix with `_ms` |
| Exact known `Flux2KleinPipeline` profiler target inside `stage_durations` | seconds, verified in frozen mixin | Multiply once by 1000; append `_ms` |
| Unsuffixed unknown stage or non-time field | Unspecified | Preserve raw; do not infer a duration |

The known profiler allowlist is `forward`, `text_encoder.forward`,
`transformer.forward`, `vae.encode`, and `vae.decode`, each prefixed with
`Flux2KleinPipeline.`. Scalars only are normalized; arrays remain in raw response
metadata. A collision between normalized names fails rather than choosing one.

For the first measured B2 request in `9723802/control-a`, raw
`stage_durations.stage_0_gen_ms` is **1290.7655239105225 ms**. The original client
incorrectly records `stage_durations.stage_0_gen_ms_ms` as
**1290765.5239105225**. The independent raw
`stage_metrics.0.stage_gen_time_ms` agrees with the corrected value.

`analyze_service.py` now records all three views:

- `server_response_fields`: original raw fields, unchanged.
- `client_recorded_server_time_metrics_ms`: original client derivation, including
  erroneous `*_ms_ms` values, unchanged.
- `corrected_server_time_metrics_ms`, `normalization_sources` and
  `normalization_differences`: offline reconstruction and explicit provenance.

`corrected_server_metric_observations` summarizes only present values and gives
sample counts and missing counts. It does not fill missing times with zero.
Metric names retain their original scope: `denoise_step_latency_ms` is not
relabeled as a directly measured transformer stage, and absent VAE/DiT profiler
fields are not fabricated.

`9723802` ended before the batch arm started because the launcher could not bind
the next port. Its analyzer report remains `INCOMPLETE_OR_INVALID`, with
`scoring_eligible=false`, no cross-arm comparisons and no latency reductions.
Only the independently validated control arm's metric observations are exposed.
That arm's corrected stage generation mean is 1287.879745 ms at B2 and
2441.645821 ms at B4. These are observations from one process, not a completed
service comparison.

`service_metric_units_fixture.json` contains a mixed-unit synthetic case and a
real first-measured B2 case tied to the original request-file SHA256.
`service_metric_units_validation.json` records fixture checks, all 19 real
requests, preservation of original bytes, and the incomplete-run scoring gate.
Existing corruption and diagnostic-hook checks also remain in their respective
local validation JSON artifacts. No GPU work was repeated for this correction.

A separate `bench_image_service_v2.py` is prepared for use after the scored job
ends and its original harness is archived. Only `time_metrics_ms()` differs
from the frozen client; byte and AST comparison confirm the HTTP timer, payload,
request loop, image checks and artifact writes are unchanged. The mixed-unit
fixture and all 19 original real response rows pass the corrected function.
`service_client_v2_local_validation.json` records both file hashes. The original
`bench_image_service.py` has not been replaced by this preparation.
