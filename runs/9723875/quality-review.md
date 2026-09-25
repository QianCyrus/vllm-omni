# Quality review: B2/B4 exact agreement; B1 remains nonexact with boundary evidence

Reviewed the completed job `9723875` (control-a, batch, control-b) and the
completed control-a process from incomplete job `9723802`. Original request
JSONL files, retained PNG hashes, decoded RGB hashes, source/model hashes and
payloads were checked. Detailed pair counts and input hashes are in
`b1-quality-audit.json`; the ordinary three-arm analyzer remains
`COMPLETE_NONEXACT_REVIEW_REQUIRED`.

## Supported scope

Across all three processes in `9723875`, every corresponding B2/B4 and alternate
prompt/seed output agrees exactly in decoded RGB hashes: **52/52 images per arm
pair**, comprising 36 measured B2/B4 images, 12 warmup images and four alternate
prompt/seed images. This is 18/18 corresponding requests. Each process's six
measured repeats per batch also agrees exactly in order. The ten retained B2/B4
and alternate-case PNGs per arm support direct decoded-RGB comparisons in the
service analyzer. These scoped comparisons pass exact equality.

The full set is **52/53 images and 18/19 requests exact** for each of the three
same-job arm pairs. The sole difference is `b1-correctness`, request **17 of 19**,
after the 16 B2/B4 warmup/measured requests and before the two alternate cases.
It is not the process's first request, and is excluded from latency scores.

## B1 numerical comparison

All values below compare retained decoded RGB uint8 arrays. MAE/RMSE/max are on
0–255 channel values; differing pixels means at least one RGB channel differs.

| Pair | MAE | RMSE | Max abs | PSNR dB | Differing pixels |
| --- | ---: | ---: | ---: | ---: | ---: |
| control-a vs batch | 0.580668 | 2.083090 | 157 | 41.756641 | 64.276218% |
| batch vs control-b | 0.542562 | 1.973495 | 157 | 42.226082 | 60.691547% |
| control-a vs control-b | 0.483484 | 1.334368 | 89 | 45.625294 | 65.049744% |

Both candidate comparisons have **larger MAE, RMSE and maximum difference** than
the same-job control-to-control pair, and lower PSNR. Consequently, B1 does not
fit an error envelope defined by those two same-job controls. A smaller fraction
of differing pixels does not reverse that conclusion.

The prior `9723802/control-a` image is byte-identical and RGB-identical to
`9723875/batch`, including B1. In fact those two processes match on **53/53
images and 19/19 requests**. Its B1 hash is
`51f303f00fd48f76faeb92d6a3f8a41e883ee1726c175a9ed7ab8d65a2f0d6e3`.
Current control-a is
`1d9ba993cd30e7333950d32165611f9417f517f319d3c29e38dadf768aa766a3`;
current control-b is recorded in `b1-quality-audit.json`.

Thus between-process baseline variation is directly observed, and a baseline
process can produce the candidate's exact output. This limits causal attribution
to the change, but does **not** establish B1 numerical noninferiority from the
same-job comparison, prove a particular cause, or satisfy an all-images exact
gate. No numerical/perceptual acceptance threshold was selected after observing
these results. B1 remains explicitly under review; no B1 speedup or full-scope
bitwise-equivalence claim is justified.

## Verified native fallback route

On source `1ccbc20901149b803a135984c2c3e4847fe981dd`:

1. Klein chooses `DistributedAutoencoderKLFlux2` for `vae_parallel_mode=batch`
   for both degree-1 controls and the degree-4 candidate.
2. `DistributedAutoencoderKLFlux2.decode` dispatches batch mode into
   `_batch_parallel_decode`.
3. `DistributedAutoencoderKL_base._batch_parallel_decode` resolves
   `native_decode = super().decode` and returns it immediately when
   `z.shape[0] <= 1`. This is before `executor.execute` and its collectives.
   Degree 1 also takes that native return for every batch size.
4. The main pipeline calls `self.vae.decode` after denoising. Its requested B1
   output count and input payload are identical across the four compared
   processes. The payload specifies seed 8137 and a CPU generator. The runner
   constructs a seeded generator when the request's generator is absent, and
   Klein passes it into `randn_tensor` during latent preparation.

Relevant paths: `models/flux2_klein/pipeline_flux2_klein.py`,
`distributed/autoencoders/autoencoder_kl_flux2.py`,
`distributed/autoencoders/autoencoder_kl.py`, and
`worker/diffusion_model_runner.py`, all beneath `vllm_omni/diffusion/`.
The captured source file hashes, model config hashes, package versions and
TP4/SP1 topology agree across the four processes. Logs confirm regional dynamic
transformer compilation on all four workers; no compile-setup failure was found.

## Initial hypotheses before the boundary diagnostic

- **Process-dependent numerical execution in the full pipeline.** The same
  degree-1 baseline produced three B1 outputs across the observed processes.
  Regional transformer compilation, TP operations, native attention and native
  VAE operations are candidate locations. The logs do not record kernel choice,
  deterministic-runtime flags, intermediate tensors or per-stage RGB causes.
  The fixed seed constrains sampling; it does not itself prove equal floating
  point execution.
- **Different prior shape/allocation history.** Before request 17, the candidate
  distributes B2/B4 into complete-image chunks of batch one, while controls use
  the native degree-1 route on the full B2/B4 tensors. That creates different
  native decoder workload history and allocation pressure even though B1 itself
  takes the same fallback. This could affect backend execution choices; no
  recorded kernel evidence proves that it did. The control processes themselves
  also disagree, so this history difference alone does not explain all results.
- **An upstream difference versus a decoder difference cannot be distinguished.**
  The artifacts contain final image hashes, not B1 pre-VAE latent hashes or
  intermediate denoising hashes. Equal raw latents were not established.

The decoded-pixel mismatch is real and is not solely PNG metadata/encoding:
independent decoding reproduces the metrics above. Existing artifacts cannot
localize the first diverging stage. The separate diagnostic job omits B1 and
cannot close this B1 quality question.

## Completed boundary diagnostic (9724061)

The separate [B1 boundary audit](../9724061/b1-diagnostic-analysis.md) now
localizes a reproduced difference before VAE decode. Both fresh processes
replayed the original 16-request B2/B4 history before the unique B1 request.
The actual request IDs, seed 8137 and four steps were verified; startup dummy
captures were excluded.

The BF16 tensors entering VAE decode differ between the two processes. In
each process, the B1 fallback output is exactly equal to an extra direct
native Diffusers decode on the same unchanged input: max absolute error, MAE
and RMSE are all zero. Source/harness/model checks and cleanup pass. The run
reports `COMPLETE_BOUNDARY_EVIDENCE` / `DIVERGENCE_PRESENT_BEFORE_VAE`.

This supports correctness of the native B1 fallback on the tested inputs. It
does not identify the first upstream operation that differs, rule out indirect
process-history effects, or prove all-image bitwise equality. The original
three-arm images and `COMPLETE_NONEXACT_REVIEW_REQUIRED` status remain intact,
along with their same-job error-envelope result. The pending boundary check
is complete; the public claim remains limited to the measured B2/B4 workload.
