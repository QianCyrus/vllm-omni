# B1 boundary diagnostic: the inputs already differ before VAE decode

Run `9724061` completed with exit 0 on the same four A100-SXM4-40GB GPUs,
source `1ccbc20901149b803a135984c2c3e4847fe981dd`, model revision and compiled
TP4/SP1 settings used by the HTTP experiment. This was a separate diagnostic;
none of its requests contribute to the reported performance scores.

Each fresh control/batch process replayed the original 16-request B2/B4 history
(two warmups and six repeats at each batch size), then the B1 check as request
17. Its prompt, CPU generator, seed 8137 and four steps match the scored run.
The hook records the rank-0 tensor just before VAE decode, the normal fallback
output, and an extra direct native Diffusers decode of that same tensor. It
returns the original fallback output. This extra execution changes diagnostic
work and must not be used as a service latency score.

The analyzer matches the real B1 capture to the HTTP request ID and server
request order. It excludes each process's startup dummy capture. Source,
harness and model identity, cleanup, tensor-file hashes, unmodified input,
output shape/dtype/finiteness, and request history all pass their recorded
checks. The hook SHA is
`2b739d98544861265c8b5c944d986935eedba61d3a05508d62c23b75eb39e54d`.

| Check | Degree-1 control | Degree-4 batch mode |
| --- | --- | --- |
| B1 fallback vs direct native decode, same input | Exact | Exact |
| Floating-output max absolute error / MAE / RMSE | 0 / 0 / 0 | 0 / 0 / 0 |
| Input shape and dtype | `[1,32,128,128]`, BF16 | `[1,32,128,128]`, BF16 |
| Input changed by either decode | No | No |

The two processes' pre-VAE tensor hashes differ:

- Control: `6d824e204f2feee2c35edd6bed8392d704ad6c1580b86d15dfc3d5eb2b79f619`.
- Batch: `4d94a7b69beb055f0ebf2726b2bad99133f0ae993b5d99c2195b92f7cb7d4926`.

Their raw VAE outputs and final RGB images also differ. The analyzer reports
`COMPLETE_BOUNDARY_EVIDENCE` and `DIVERGENCE_PRESENT_BEFORE_VAE`, with no audit
errors. Thus this replay's cross-process difference is already present before
VAE decode, and the new B1 fallback matches direct native decode for each
observed input. It does not establish which upstream operation first differs,
or rule out indirect effects of prior process/allocation history.

The earlier scored run remains `COMPLETE_NONEXACT_REVIEW_REQUIRED`: its actual
B1 images are not rewritten or relabeled exact. This diagnostic does not prove
bitwise equivalence for the whole compiled pipeline, identify the cause of
every historical B1 output, establish a general perceptual-quality threshold,
or measure a B1 speedup. No acceptance threshold was chosen after the result.
The exact B2/B4 comparison and multi-image latency claim remain separately
scoped. See [the full JSON audit](b1-diagnostic-analysis.json) for request IDs,
captures, SHA checks and excluded startup records.
