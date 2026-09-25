# Phase-separated diagnostic

Run 9718466; source `ed5fc2b83b5dcfda6bae7132b4ac87a4b2ca54c2`.

Timings below are diagnostic CUDA-event measurements with instrumentation. They are not scored latency.

| Rank | Startup dummy, sum ms | Request warmup, sum ms | Measured 4-step sum ms | Measured step mean ± sample std, ms |
| --- | ---: | ---: | ---: | ---: |
| 0 | 4252.158 | 722.551 | 622.759 | 155.690 ± 1.325 |
| 1 | 4252.121 | 724.342 | 619.638 | 154.909 ± 1.578 |
| 2 | 4251.839 | 725.215 | 628.581 | 157.145 ± 4.168 |
| 3 | 4251.815 | 725.524 | 623.870 | 155.967 ± 3.566 |

Forward indices: startup 1–2; request warmup 3–6; measured request 7–10. Each forward has 25 target pre-attention calls per rank.

The existing analyzer's all-forward average includes startup and warmup. Use this separated report for stage attribution.

Observed B2 Q/K/V shape before gathering: `[2,1536,24,128]`; gathered K/V: `[2,6144,24,128]`. Joint tensors are absent. Image tokens are sharded, while replicated text is concatenated before gathering in this native model path.

This confirms the target optimization runs. It supports a matched native base/head comparison; it does not establish single-GPU model equivalence or the separate joint-text path.

Measured request pipeline profiler (seconds; rank 0 encoder/VAE from structured results, other values from rounded logs):

| Rank | Pipeline forward | Text encoder | VAE decode |
| --- | ---: | ---: | ---: |
| 0 | 1.058963 | 0.042667 | 0.375345 |
| 1 | 1.059358 | 0.044212 | unavailable |
| 2 | 1.061146 | 0.040337 | 0.375390 |
| 3 | 1.059949 | 0.040812 | 0.375718 |

Transformer, encoder and VAE measurements are nested inside pipeline forward. Do not add them to the pipeline total.

Transformer CUDA events include observer overhead and host-induced gaps. These are not uninstrumented performance scores.
Transformer, text encoder and VAE timings are nested inside pipeline forward; never add them to that total.
Representatives are deduplicated by tensor metadata across all calls; first-seen B2 shapes occur during request warmup.
Observed B2 native path gathers sequence 1536 to 6144, with no joint tensors. It does not validate the separate joint-text path.
A single measured diagnostic request is attribution evidence, not a speedup estimate or an independent quality comparison.

Artifact files checked: 23. Verification problems: 0.

Skipped ambiguous interleaved profiler prefixes at line 518: Flux2KleinPipeline.vae.encode
Skipped ambiguous interleaved profiler prefixes at line 549: Flux2KleinPipeline.vae.decode
Omitted incomplete optional profiler series: rank 0, Flux2KleinPipeline.vae.decode
Omitted incomplete optional profiler series: rank 1, Flux2KleinPipeline.vae.decode