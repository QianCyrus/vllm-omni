# Diagnostic stage comparison

One instrumented measured request per arm, rank 0. This is stage attribution, not a new scored speedup estimate.

Base: `ed5fc2b83b5dcfda6bae7132b4ac87a4b2ca54c2` (run 9718466). Head: `99c27812387ab78ed26d354bc10df8595fb173d8` (run 9718248).

| Stage or enclosing total | Base, ms | Head, ms |
| --- | ---: | ---: |
| Transformer, four steps (CUDA events) | 622.759 | 594.786 |
| Text encoder | 42.667 | 40.403 |
| VAE decode | 375.345 | 375.870 |
| Pipeline forward (enclosing total) | 1058.963 | 1028.383 |
| Request wall (enclosing total) | 1145.348 | 1121.908 |

Encoder and VAE values come from structured measured results. Pipeline forward comes from validated rank-0 log lines. Transformer timings include diagnostic observer overhead.

Both runs: ranks 0–3; 25 target pre-attention calls per transformer forward; 50 startup, 100 warmup and 100 measured calls per rank. B2 Q/K/V are `[2,1536,24,128]`, gathered K/V are `[2,6144,24,128]`, and joint tensors are absent. Source hashes and artifact checks passed.

- Startup forwards 1–2 and request warmup forwards 3–6 are excluded from every stage value in this comparison.
- Transformer, encoder and VAE timings are nested inside pipeline forward. Never add them to that enclosing total.
- The native model gathers replicated text together with sharded image tokens; this does not validate the separate joint-text path or single-GPU equivalence.
- Base shutdown logged no terminate or kill escalation. Head shutdown required terminate after the 15-second graceful wait; both Omni.close calls returned normally.
- cleanup_complete means close returned without exception; the driver did not independently inspect worker liveness.

Teardown implementations are byte-identical between the two source snapshots. The cleaner waits 15 seconds, then terminates surviving workers; it logs rather than raises if cleanup remains incomplete. Neither run logged kill or incomplete-cleanup warnings.
