# Full precheck closure: VAE batch parallel decode

This is an addendum to `precheck-full.md`; that report remains unchanged as a
record of the earlier review. The code is still the reviewed and measured
candidate `1ccbc20901149b803a135984c2c3e4847fe981dd`.

**Code/evidence verdict: ready for a scoped performance PR, with the limits
below stated in its body.** The earlier full-model and reproduction gaps are
closed by completed runs and the checked-in evidence package. This is not an
all-output exactness or full-repository CI pass.

- HTTP job `9723875` completed the same-source degree-1/4/1 comparison with
  clean exits, source/harness checks, two warmups and six samples per B2/B4
  cell. Latency fell 15.42–15.91% at B2 and 19.76–19.91% at B4. Mean and sample
  standard deviation are retained for each arm; no control pooling or dropped
  samples. All 52 corresponding B2/B4 images per arm pair are exact.
- Separate job `9723970` attributes the gain to VAE decode: about 220/540 ms
  less at B2/B4 while DiT time stays close. Three warmed samples per cell use
  synchronized host wall measurements, including communication. Interleaved
  worker logs limit rank/call alignment; this is disclosed and no rank maximum
  or exact cross-run latency decomposition is inferred.
- The original unscored B1 images differ and remain nonexact in the audit.
  Job `9724061` repeats the original request history and finds different input
  tensors before VAE decode. Each arm's fallback matches direct native decode
  on its own unchanged input exactly. This closes the requested boundary
  investigation; it does not establish universal bitwise equivalence or the
  first upstream source of numerical variation.
- The evidence package includes runnable component/HTTP scripts, portable
  commands, frozen per-run harnesses, source/model hashes, all raw measured
  samples, selected PNGs, all-image hashes, B1 input tensors, diagnostic
  captures, and both failed launch attempts. No model weights are included.
  The code PR stays small; the evidence branch carries the benchmark material.
- Validation remains 42 focused CPU tests passed, 15 warnings. Applicable
  non-mypy pre-commit hooks passed. Mypy has 32 errors on exact base and 32 on
  head, zero new errors; it was checked separately then skipped in the combined
  invocation. Full-repository L1/L2 and hosted CI were not run locally.
- Live PR and performance issue templates were rechecked. Three author PRs
  were open before this submission, below the five-PR cap. The draft has the
  required title/sections, hardware and software versions, model/source SHAs,
  Bottleneck/Value/A-B evidence, and a Codex disclosure. The combined issue
  search/chatbot checkbox stays unchecked because the chatbot was unavailable.

Publication mechanics remain: commit/push the evidence, replace its immutable
URL and the new issue number, then create the issue and linked PR. These steps are already authorized by the user;
this note requests no new approval. The public author-side check comment must
name AI assistance and must not invent personal human review.

Current-main compatibility was checked against
`a8576ccb725c4e21cd13c3eb5f9a546b21149d2b`: read-only merge-tree exited 0
with tree `991b0dc314f64effb2f2151bc17b4a588e31a772`. Later upstream changes
are in AR/duplex paths, with no diffusion overlap. The tested head is retained.
