# Full-model A/B results

Each run is reported separately. A/B/A baselines are not pooled. Positive reduction means lower head latency; negative means slower. Spread is across measured requests within a run, not independent job repetitions.

| Run | Role | Kind | Validation | Mean ± std, s | Median | Min–max |
|---|---|---|---|---:|---:|---:|
| runs/9718337 | base | scored | PASS | 1.128337 ± 0.015133 | 1.125502 | 1.115611–1.155189 |
| runs/9718379 | head | scored | PASS | 1.094734 ± 0.004220 | 1.094863 | 1.089789–1.101289 |
| runs/9718423 | base | scored | PASS | 1.120519 ± 0.005266 | 1.120059 | 1.115104–1.127351 |
| runs/9718248 | head | diagnostic | PASS | 1.121908 ± 0.000000 | 1.121908 | 1.121908–1.121908 |
| runs/9718466 | base | diagnostic | PASS | 1.145348 ± 0.000000 | 1.145348 | 1.145348–1.145348 |

## Head versus each baseline

| Baseline | Head | Config/completion check | Mean latency reduction | Output check |
|---|---|---|---:|---|
| runs/9718337 | runs/9718379 | True | +2.98% | PIXEL_HASH_MATCH |
| runs/9718423 | runs/9718379 | True | +2.30% | PIXEL_HASH_MATCH |

A timing result does not establish output equivalence. Missing images, PNG hash failures, and pixel hash differences leave strict equivalence unverified. General image quality is not assessed. Config differences and optional pixel metrics are in `summary.json`.

## Diagnostic route and stages

Transformer timings are instrumented and nested inside pipeline forward. Do not add them to pipeline totals or use them as scored latency. Hook counts include startup and warmup. `stage_durations_raw` retains original names and mixed units; no stage values are blindly summed.

**runs/9718248**: RECORDED

| Rank | AllGather-KV preparation calls | Transformer calls | CUDA time mean per observed forward, ms | Source hashes match |
|---|---:|---:|---:|---|
| 0 | 250 | 10 | 681.4634231567383 | True |
| 1 | 250 | 10 | 680.4280464172364 | True |
| 2 | 250 | 10 | 681.5895462036133 | True |
| 3 | 250 | 10 | 680.0607192993164 | True |
| None | 0 | 0 | — | True |

**runs/9718466**: RECORDED

| Rank | AllGather-KV preparation calls | Transformer calls | CUDA time mean per observed forward, ms | Source hashes match |
|---|---:|---:|---:|---|
| 0 | 250 | 10 | 559.7467514038086 | True |
| 1 | 250 | 10 | 559.6100708007813 | True |
| 2 | 250 | 10 | 560.5635879516601 | True |
| 3 | 250 | 10 | 560.1209503173828 | True |
| None | 0 | 0 | — | True |
