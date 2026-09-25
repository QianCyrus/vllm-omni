# Partial control run; failed whole-job comparison

The first control arm completed 19 HTTP requests: 12 timed requests, four
warmups, one B1 correctness request, and two alternate prompt/seed requests.
The server exited cleanly with return code zero after SIGINT; its process group
and the client process group were gone. Source verification passed.

The next arm failed before server spawn at the launcher's `probe.bind`, with
`OSError: [Errno 98] Address already in use`. The whole job exited 1. There is
no candidate or second-control result, so this run does not support an A/B
speedup claim. The port preflight occurred before the per-arm output directory
and exception handler; its failure is recorded in `job.log`.

The original launcher lacked `SO_REUSEADDR` on its probe. A separate Linux
loopback test reproduced a plain-bind failure after clean server shutdown,
verified that `SO_REUSEADDR` permits reuse, and verified that it still rejects
a live listener. See the current `validation/port_preflight_validation.json`.
The retry changes only that probe option and explanatory comments.

The frozen client's derived `stage_durations.*_ms_ms` fields incorrectly apply
an extra factor of 1000 to values already in milliseconds. Original values in
`server_response_fields.metrics.stage_durations` and HTTP `wall_ms` are
preserved. Correct them in offline analysis, not by editing these raw records.

`harness/` holds the exact files used in this attempt. PNGs are generated
Klein4B outputs; prompts, seeds and generation settings are in
`control-a/http/requests.jsonl`. They are validation artifacts, not ground truth.
