# Phase-separated diagnostic

Run 9718248; source `99c27812387ab78ed26d354bc10df8595fb173d8`.

Timings below are diagnostic CUDA-event measurements with instrumentation. They are not scored latency.

| Rank | Startup dummy, sum ms | Request warmup, sum ms | Measured 4-step sum ms | Measured step mean ± sample std, ms |
| --- | ---: | ---: | ---: | ---: |
| 0 | 5445.121 | 774.727 | 594.786 | 148.697 ± 5.240 |
| 1 | 5444.829 | 770.440 | 589.011 | 147.253 ± 3.203 |
| 2 | 5445.457 | 775.721 | 594.718 | 148.679 ± 4.938 |
| 3 | 5444.971 | 771.926 | 583.710 | 145.927 ± 0.772 |

Forward indices: startup 1–2; request warmup 3–6; measured request 7–10. Each forward has 25 target pre-attention calls per rank.

The existing analyzer's all-forward average includes startup and warmup. Use this separated report for stage attribution.

Observed B2 Q/K/V shape before gathering: `[2,1536,24,128]`; gathered K/V: `[2,6144,24,128]`. Joint tensors are absent. Image tokens are sharded, while replicated text is concatenated before gathering in this native model path.

This confirms the target optimization runs. It supports a matched native base/head comparison; it does not establish single-GPU model equivalence or the separate joint-text path.

Measured request pipeline profiler (seconds; rank 0 encoder/VAE from structured results, other values from rounded logs):

| Rank | Pipeline forward | Text encoder | VAE decode |
| --- | ---: | ---: | ---: |
| 0 | 1.028383 | 0.040403 | 0.375870 |
| 1 | 1.029405 | 0.042410 | 0.376421 |
| 2 | 1.026593 | 0.040675 | 0.376029 |
| 3 | 1.027606 | 0.047233 | 0.376379 |

Transformer, encoder and VAE measurements are nested inside pipeline forward. Do not add them to the pipeline total.

Transformer CUDA events include observer overhead and host-induced gaps. These are not uninstrumented performance scores.
Transformer, text encoder and VAE timings are nested inside pipeline forward; never add them to that total.
Representatives are deduplicated by tensor metadata across all calls; first-seen B2 shapes occur during request warmup.
Observed B2 native path gathers sequence 1536 to 6144, with no joint tensors. It does not validate the separate joint-text path.
A single measured diagnostic request is attribution evidence, not a speedup estimate or an independent quality comparison.

Artifact files checked: 23. Verification problems: 0.
