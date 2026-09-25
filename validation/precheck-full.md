# Full PR precheck: VAE batch parallel decode

Review date: 2026-09-26. Read-only review; no source changes, commits, pushes,
GitHub posts, or GPU jobs were made by this review. This report is the only
new file written for the precheck.

- Worktree: `/Users/aniya/vllm-omni-worktrees/vae-batch-parallel`.
- Branch: `codex/vae-batch-parallel`.
- Reviewed head: `1ccbc20901149b803a135984c2c3e4847fe981dd`.
- Merge base: `43e507117f04a86f2df8d405cbe03c1b1642eac9`.
- Inspected upstream main: `5d0963b3dc325475730c1b1cc8321198da98bb39`.
- Intended public title: `[Core] Add batch parallel VAE decode`.
- Mode: **full**; types: performance, existing diffusion feature, general.

## Verdict

**No high-confidence code correctness blocker found in the reviewed diff.**
There are **two outstanding publication/readiness items** and **two
nonblocking code/check caveats** below. This is not a full-ready PASS while
the new model/HTTP evidence and public reproduction material are incomplete.

The running full-model/HTTP job `9723702` was reported pending by the root
agent. No completed local result for it was available to this review. Its
future success must not be inferred from the component results.

| Dimension | Result | Evidence |
| --- | --- | --- |
| PR title | PASS | Intended public title uses documented `[Core]`; historical commit title is not used as the PR title. |
| Scope and description | PENDING | Skeleton accurately describes the diff and its limits, but final model/HTTP results, issue number, and commands remain editor slots. |
| Base and merge compatibility | PASS at inspected snapshot | Main is one AR-only commit ahead; changed paths do not overlap; read-only merge-tree has no conflict markers. |
| DCO | PASS | Candidate commit contains `Signed-off-by: QianCyrus ...`. |
| Code quality | WARNING | One new forwarding signature uses `Any` and has no explicit return type; no acute pattern blocker. |
| Examples policy | PASS | No Python examples added, copied, or renamed. |
| Simplification | PASS | No strong simplification or dead-code blocker found; existing executor is reused. |
| Configuration and registry | PASS by inspection and focused tests | Mode reaches decoder; unsupported VAE classes and incompatible DP/PP/CFG topologies are rejected. |
| CPU behavior | PASS, reviewed prior execution | `42 passed, 15 warnings in 42.17s`; focused suite, not whole repo. |
| Real-weight component correctness | PASS, scoped | B1/B2/B4/B5 outputs agree exactly on all four A100 ranks; dtype, shape, finiteness and cleanup checks pass. |
| Component performance evidence | PASS, scoped | Hardware, versions, A/B order, warmups, samples, spread, memory and neutral B1 case recorded. |
| Full-model/HTTP performance and quality | PENDING | Job `9723702` unfinished/unavailable at review; no result claimed. |
| Public benchmark reproduction | INCOMPLETE | Scripts remain in the experiment archive; HEAD contains no checked-in runnable performance benchmark. |
| Local lint gates | PASS with caveat | Applicable non-mypy hooks pass; mypy has 32 unchanged base errors and zero new errors. |
| Full repository L1/L2 and hosted CI | NOT RUN / NOT AVAILABLE | Do not claim them from the focused CPU tests or component job. |

## Items to finish before the authorized publication

1. **Finish the actual same-workload model/HTTP A/B and quality gate.**
   Record the final source hash, four-worker topology, image count, resolution,
   seed/prompt, steps, dtype, compile settings, warmups, repeats and sample
   spread. Keep component, native generation and HTTP timings separate.
   Compare ordered generated outputs and baseline repeat variance. Report
   failures or neutral cases as well as wins. The user's publication condition
   includes these experiments; a component-only result does not satisfy it.
2. **Make reproduction reviewable and finalize the public body.**
   The repository full performance checklist asks for a checked-in runnable
   benchmark and an exact command. `bench_vae_distributed.py`,
   `launch_model_service.py`, and `bench_image_service.py` currently exist only
   in the experiment archive. Add an appropriate scoped benchmark/reproduction
   artifact, link the evidence, include exact executed commands, and replace
   the issue-number and evidence placeholders. Recheck any added source files.

The issue form also contains a required checkbox that combines prior-report
search with asking the documentation chatbot. Search was performed; chatbot
completion was not verified by this review. Do not check that box falsely.
This records the form requirement, not a request for another user approval.

## Nonblocking caveats

- `vllm_omni/diffusion/distributed/autoencoders/autoencoder_kl.py:60` adds
  `_batch_parallel_decode(..., *args: Any, **kwargs: Any)` without an explicit
  return annotation. This is a warning under the skill's typing rule. The
  method preserves the native Diffusers arguments, performs no kwargs string
  lookup, and returns either `DecoderOutput` or a one-tensor tuple. An explicit
  return union would improve the contract; changing the forwarding interface
  is not required to establish correctness here.
- Mypy is **not green**: the initial head had 34 errors; two new mixin/MRO
  errors were addressed with targeted annotations. The final comparison is
  32 errors on exact base and 32 on head, zero new errors. The final combined
  pre-commit run skipped mypy after this separate comparison. Preserve this
  wording in the public evidence instead of claiming all hooks passed.

## Correctness and ownership review

The new mode is optional; the default remains `tile`. The public mode is
accepted by CLI and both configuration representations, forwarded by existing
configuration plumbing, checked by the registry, and stored in the existing
executor. Klein chooses the distributed Flux2 VAE only for batch mode.

`batch_split` caps task count by batch length, configured degree, and existing
worker count. It uses contiguous tensor splits along dimension zero; complete
images retain their spatial context. The existing workload balancer may
assign tasks to ranks, while image coordinates preserve the original ordering
during `batch_merge`.

`_batch_parallel_decode` resolves the native decoder through the class MRO
before defining the task callback, so each task does not recursively enter
the distributed wrapper. The callback preserves native arguments and native
slicing/tiling within its assigned images. The spatial-enable predicate is
false in batch mode, which keeps the Flux2 encode path native and avoids
spatial executor recursion.

Idle ranks still enter the existing executor's shape reduction, metadata/data
gather, and final result broadcasts. The output dtype is specified even for
ranks with no local images. B1, degree one, and uninitialized distributed
execution use native decode. The WORLD-based path rejects DP, PP and CFG
degrees above one at configuration boundaries and at active runtime decode;
inferred DP is checked after world-size resolution.

The existing executor owns task balancing, result packing, group communication,
merge and broadcast. The PR adds no separate process group, allocator, request
batcher, cache, mutable request state, or new dependency. The two configuration
checks are intentional checks at distinct public construction/materialization
boundaries, not evidence that one can safely be deleted. Replacing gather and
broadcast with a new communication path would require separate performance
and compatibility evidence and is outside this small feature.

Inspection found no newly dead methods, unused production imports,
unreachable fallback, swallowed exception, model-specific Python example,
new library `torch.cuda` helper, or allowlist expansion. The added
`weight.copy_()` is test-only initialization of three scalar weights, not a
production hot-path copy. No new asynchronous blocking or lock was added.

## Test coverage and limits

The new CPU module uses real Gloo collectives in four spawned processes and
native AutoencoderKL/Flux2 decode methods with a small deterministic decoder.
It covers changing B2/B5/B1 inputs, idle ranks, order, both return formats,
FP32/BF16/autocast, native argument forwarding, degree-one slicing fallback,
no-distributed fallback, configuration guards, tiling/slicing flags, unsupported
VAE rejection, and Klein factory selection. It uses `core_model`, `cpu`,
`diffusion` and `parallel` marks. The ready CI's existing
`tests/diffusion -m 'core_model and cpu'` discovery includes it.

Component job `9723523` uses the real pretrained Klein VAE, BF16, four
A100-SXM4-40GB GPUs with NVLink, and fixed synthetic latents. All four ranks
report max absolute error, MAE and RMSE zero for B1/B2/B4/B5. The job exited
zero and each rank records cleanup completion. It measures the whole VAE
component path including communication, not a pure kernel or full request.

The reviewed component means are 380.572 → 157.172 ms at B2,
696.512 → 155.839 ms at B4, and 857.028 → 377.247 ms at B5. B1 is
153.659 → 153.717 ms, a neutral result within the observed variation.
Full-model memory, single-image speedup, other models/backends, real-weight
native tiling/slicing combinations, and model-wide exact numerical equivalence
are not established by those measurements.

New-top-level-model requirements such as a new registry entry, a model support
table row, a new pipeline `load_weights`, and Cache-DiT integration are not
applicable: this is an execution option for existing VAE wrappers and an
existing pipeline. Full generation-path validation remains applicable and
pending. No service throughput or concurrency-scaling claim is made.

## Verification performed in this review

- Read the full precheck workflow, performance/general/diffusion checklists,
  code-quality and examples references, and the simplification skill.
- Read the architecture overview, active VAE design, diffusion parallelism
  contract, current user-facing VAE guide, changed production/tests, relevant
  callers, and test-marker/CI routing guidance.
- `git merge-base HEAD origin/main` returned the recorded base. The local
  tracking ref was still at that base, so current-main compatibility was
  separately checked against the explicit `5d0963b3...` commit object.
- `git log BASE..MAIN` shows one commit: AR runner CUDA Graph payload slicing
  (#8159). Only `vllm_omni/worker_v2/omni_ar_model_runner.py` and its test changed;
  there is no path overlap with this PR's 11 files.
- `git merge-tree BASE MAIN HEAD` completed without conflict markers. No merge
  or rebase was performed, preserving the measured source snapshot.
- `git diff --check BASE..HEAD` passed; the worktree was clean.
- Scanned added Python lines for the five required fragility patterns and
  manually classified every hit.
- Compared current files against `service_source_manifest.json`: no hash
  mismatches. Compared ten Python files in the archived component overlay
  against current source ASTs: no mismatches.
- Read the focused CPU log, component metadata/summary and per-rank correctness
  and cleanup files, pre-commit log, and mypy base/head comparison. Existing
  tests were not rerun merely for this read-only review.

Useful focused commands for a reviewer with the repository's Python/vLLM,
Diffusers, pytest and pytest-mock dependencies installed (CPU Gloo required):

```bash
pytest -q -o addopts='' \
  tests/diffusion/distributed/test_vae_batch_parallel.py \
  tests/diffusion/distributed/test_distributed_vae_executor.py

# Existing CI discovery lane; broader than the focused tests above.
pytest -sv tests/diffusion -m 'core_model and cpu'
```

These commands describe focused/broader reproduction lanes; this review did
not run them or claim that the two-file command alone produced the archived
42-test total. Final public validation must include the exact command used
for the reported result.

## Policy and final publication notes

AI disclosure must name Codex and its role in implementation, tests,
benchmarks, analysis and writing. Do not fabricate a contributor's personal
review statement. The new five-open-PR cap was checked separately: QianCyrus
had three open PRs, so this VAE PR is within the limit at that snapshot.

The user has already authorized the new issue and linked PR once the agreed
experiments pass. This report does not introduce a new permission gate.
Re-evaluate the pending items when the root agent has final results, and
update the evidence/source hashes if implementation changes.
