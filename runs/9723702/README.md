# Failed first HTTP attempt

This run failed before API readiness, with process exit code 1. There are no
completed HTTP benchmark samples and no valid serving performance result.

The server tried to create its speaker cache under the job's unwritable default
home directory. The traceback is preserved in `control-a/server.log`; the
launcher failure is in `control-a/failure.json`. Source verification and process
cleanup succeeded, but those checks do not turn the server run into a pass.

`harness/` contains the exact launcher, client and PJM wrapper used by this
failed attempt. Their hashes match `harness_before.sha256`. The corrected
current harness elsewhere in the repository is a separate snapshot and must
not replace these files.

The retry places `SPEAKER_SAMPLES_DIR` under `/tmp`. Retry results are pending.
