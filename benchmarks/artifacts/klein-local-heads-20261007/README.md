# Klein local-head tensor parallelism: native inference evidence

[Download the reproduction package](klein-native-evidence.zip). Extract it and read `fresh-evidence/README.md`; paths below refer to the extracted package.

For the measured **four-image request**, the change lowers native HTTP latency by **21.81%**, from a two-baseline mean of **2685.93 ms** to **2100.21 ms**, on **4 × NVIDIA A100-SXM4-40GB**. The repeated B4 workload passes the output and timing checks described below. B1 does not pass the cross-process output check and is excluded from the performance claim.

## Workload and sources

- Model: `black-forest-labs/FLUX.2-klein-4B`, revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`.
- Base O: `5f95115e703ffe51e03e40ae23c78ff841c3335e`; candidate A: `39a06eac411a377e0bc1eb13bdbceb10c9783bf6`.
- BF16, TP4/SP1, `TORCH_SDPA`, regional compilation with dynamic shapes; the pipeline's native startup warmup is unchanged. Cache backend and VAE fast path are off; VAE tile parallel degree is 1.
- PyTorch `2.13.0+cu130`, vLLM `0.31.0`, diffusers `0.40.0`, transformers `5.14.1`. The imported vLLM-Omni source is bound to the commits above; its installed distribution label is `0.30.0rc1+kvbench`.
- Images API, 1024 × 1024, four steps, guidance 1, concurrency 1, seed 8137, CPU generator. B1 and B4 mean one and four images in one request.
- Fixed prompt: “A small red sailboat on a calm blue lake, distant green hills, soft morning light, realistic photograph.”
- Three fresh services run O1 → A → O2 on the same GPUs, with fresh compiler caches. Each receives five warmup and 30 measured requests per batch. The latency run has no profiler or forward instrumentation; startup receipts check source, worker configuration, local-head setup and compiled-block wrappers.

## Native HTTP latency

| B4 arm | Mean ± sample SD (ms) | Median (ms) | p95 (ms) |
| --- | ---: | ---: | ---: |
| O1 | 2679.09 ± 5.25 | 2679.04 | 2686.58 |
| A | 2100.21 ± 5.90 | 2100.78 | 2110.43 |
| O2 | 2692.77 ± 73.13 | 2677.21 | 2715.92 |

Reduction is `1 - mean(A) / ((mean(O1) + mean(O2)) / 2)`. Baseline mean drift is **0.51%**, below the predeclared 3% limit. Every measured request is included, including O2's slower samples. These are 30 requests within each service, not 30 independent service restarts; the table gives descriptive statistics, not a process-level confidence interval.

All **420 recorded B4 images**—three arms × 35 requests × four images—match exactly at each image index, including warmups. PNG file and decoded RGB hashes were recomputed from the saved images. These are repeated outputs for one fixed prompt and seed, not 420 diverse quality samples. This check does not establish hidden-tensor equality, broader image quality or future process determinism.

For **B1**, A and O2 match, while O1 differs from both on all 35 recorded requests. Each service is internally consistent. The baseline also changes across restarts, so this run does not isolate the source of the B1 difference. B1 timings and failed qualification remain in `comparison.json`; no B1 model-level speedup is claimed.

## Stage attribution

The separate B4 diagnostic locates the reduction in the transformer: its four accumulated calls fall from **1519.92 ms to 943.81 ms**, a **37.90%** reduction. Text encoder and VAE durations remain close.

| B4 stage | O mean ± sample SD (ms) | A mean ± sample SD (ms) | Change |
| --- | ---: | ---: | ---: |
| Text encoder | 40.391 ± 0.033 | 40.066 ± 0.146 | −0.325 ms |
| Transformer, four calls accumulated | 1519.915 ± 0.659 | 943.808 ± 1.917 | −576.107 ms (−37.90%) |
| VAE decode | 694.899 ± 0.693 | 694.426 ± 0.618 | −0.473 ms |

The separate diagnostic uses one O and one A service, three warmups and five measured requests per batch. The existing pipeline profiler times `text_encoder.forward`, `transformer.forward` and `vae.decode`, synchronizing before and after each outer call. The response contains rank 0's durations, not a sum or maximum across GPUs. Regional block compilation is retained. These instrumented timings attribute time to stages and do not replace the native HTTP measurements above.

All 32 recorded B4 diagnostic images per arm match at each image index across O/A and the same-variant native services, including both native baselines. B1 diagnostic output comparisons fail; its raw results are retained without a qualified B1 stage-reduction claim. Matching B4 outputs does not prove instrumentation leaves timing unchanged.

The frozen stage protocol contains several inherited native-run prose fields. `protocol/stage-protocol-clarification.json` identifies them and records the actual O/A, three-warmup/five-measurement design enforced by the launcher and request records; the original protocol is preserved unchanged.

## Mechanism and reproduction

Klein-4B has 20 single-stream blocks. At TP4, each GPU now computes six of the 24 attention heads and 2304 of the 9216 MLP channels. Input and output projection GEMM dimensions remain unchanged. Attention and MLP results are gathered after their local computation; logical gathered feature width falls from 10D to 5D, while the collective count rises from two to three per block. This is a 50% reduction in logical gathered volume, not a claim of 50% lower communication time.

`native_harness/` and `stage_harness/` contain the frozen launchers, workload client and integrity checks. Their READMEs describe the flags and separate timing/diagnostic modes. Use clean source directories at the two commits, a local model snapshot at the recorded revision, and a compatible GPU environment. Check each `BUNDLE_MANIFEST.json` before and after a new run. The native sequence is O1/A/O2 with `--batches 1 4 --warmup 5 --repeats 30 --steps 4 --compile-mode regional`; the stage commands use `--warmup 3 --repeats 5 --diagnostic --stage-metrics`.

`comparison.json` holds per-batch gates, all native latency samples, output hashes and source/protocol bindings. `stage-comparison.json` holds the separate diagnostic, with per-batch output checks in `stage-batch-checks.json`. Per-arm metadata, request records, source manifests and startup receipts remain available under `native_run/` and `stage_run/`. The [earlier component ablations](https://github.com/QianCyrus/vllm-omni/tree/f882d97a627d406e5056c47936818930cd6ce5a3/benchmarks/artifacts/klein-local-heads-20261006) provide the O/L/H/M/A block-level experiment.

Private machine paths use explicit placeholders; host names and GPU UUIDs are redacted. Numeric values, content hashes and qualification flags are unchanged. Original receipt hashes refer to the original records; `sanitization-manifest.json` also records the public copies' hashes. `SHA256SUMS` covers the bundle. Images, model weights, logs and compiler caches are omitted; full image revalidation requires the saved originals or a new run. The included comparator cannot rehash omitted ONGs from their hashes alone.
