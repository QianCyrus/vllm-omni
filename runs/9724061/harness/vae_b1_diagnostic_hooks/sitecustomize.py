"""Opt-in rank-0 B1 VAE boundary probe; no early Torch or model imports.

Set OMNI_VAE_B1_DIAGNOSTIC=1 and OMNI_VAE_B1_DIAGNOSTIC_DIR to a fresh arm-local
artifact directory, and use the launcher's --diagnostic/--diagnostic-hook-dir.
Every B1 capture includes one EXTRA direct native decode. Never score timings.
"""

import os

if os.environ.get("OMNI_VAE_B1_DIAGNOSTIC") == "1":
    import contextvars
    import functools
    import hashlib
    import importlib.abc
    import importlib.machinery
    import itertools
    import json
    import sys
    import time
    from pathlib import Path

    _VAE = "vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_flux2"
    _PIPELINE = "vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein"
    _CONTEXT = contextvars.ContextVar("vae_b1_request", default=None)
    _COUNTER = itertools.count()

    def _hash(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _cpu(tensor):
        return tensor.detach().to(device="cpu", copy=True).contiguous()

    def _tensor_info(cpu_tensor):
        import torch

        return {
            "shape": list(cpu_tensor.shape),
            "dtype": str(cpu_tensor.dtype),
            "sha256_contiguous_bytes": hashlib.sha256(cpu_tensor.view(torch.uint8).numpy().tobytes()).hexdigest(),
        }

    def _sample(value):
        return value.sample if hasattr(value, "sample") else value[0]

    def _install_pipeline(module):
        cls = module.Flux2KleinPipeline
        original = cls.forward

        @functools.wraps(original)
        def forward(self, req, *args, **kwargs):
            sampling = req.sampling_params
            token = _CONTEXT.set(
                {
                    "request_id": req.request_id,
                    "request_ids": list(req.request_ids),
                    "seed": sampling.seed,
                    "num_outputs_per_prompt": sampling.num_outputs_per_prompt,
                    "num_inference_steps": sampling.num_inference_steps,
                    "generator_device": str(sampling.generator_device),
                }
            )
            try:
                return original(self, req, *args, **kwargs)
            finally:
                _CONTEXT.reset(token)

        cls.forward = forward

    def _install_vae(module):
        cls = module.DistributedAutoencoderKLFlux2
        original = cls._batch_parallel_decode
        direct_native_decode = module.AutoencoderKLFlux2.decode

        @functools.wraps(original)
        def decode(self, z, return_dict=True, *args, **kwargs):
            import torch
            import torch.distributed as dist

            rank = dist.get_rank() if dist.is_initialized() else 0
            if z.shape[0] != 1 or rank != 0:
                return original(self, z, return_dict, *args, **kwargs)
            directory = os.environ.get("OMNI_VAE_B1_DIAGNOSTIC_DIR")
            if not directory:
                raise RuntimeError("B1 diagnostic requires OMNI_VAE_B1_DIAGNOSTIC_DIR")
            directory = Path(directory).resolve()
            directory.mkdir(parents=True, exist_ok=True)
            capture_id = f"pid{os.getpid()}-b1-{next(_COUNTER):03}"
            before = _cpu(z)
            record = {
                "capture_id": capture_id,
                "pid": os.getpid(),
                "rank": rank,
                "scored": False,
                "started_unix_s": time.time(),
                "request": _CONTEXT.get(),
                "hook_sha256": _hash(Path(__file__)),
                "input": _tensor_info(before),
                "input_device": str(z.device),
                "input_stride": list(z.stride()),
                "vae_parallel_degree": self.distributed_executor.parallel_size,
                "vae_world_size": self.distributed_executor.world_size,
                "scope": "B1 fallback plus an extra direct native decode on the same unchanged tensor; diagnostic only",
            }
            try:
                if os.environ.get("OMNI_VAE_B1_SAVE_TENSORS", "1") == "1":
                    latent_path = directory / f"{capture_id}-input.pt"
                    torch.save(before, latent_path)
                    record["input_tensor_file"] = latent_path.name
                    record["input_tensor_file_sha256"] = _hash(latent_path)
                result = original(self, z, return_dict, *args, **kwargs)
                fallback = _cpu(_sample(result))
                record["fallback_output"] = _tensor_info(fallback)
                record["input_after_fallback"] = _tensor_info(_cpu(z))
                if record["input"] != record["input_after_fallback"]:
                    raise RuntimeError("Native fallback mutated the input; same-input direct comparison aborted")
                direct_result = direct_native_decode(self, z, return_dict, *args, **kwargs)
                direct = _cpu(_sample(direct_result))
                record["direct_native_output"] = _tensor_info(direct)
                record["input_after_direct"] = _tensor_info(_cpu(z))
                if record["input"] != record["input_after_direct"]:
                    raise RuntimeError("Direct native decode mutated the input")
                if fallback.shape != direct.shape or fallback.dtype != direct.dtype:
                    raise RuntimeError("Fallback/direct output shape or dtype differs")
                delta = fallback.float() - direct.float()
                if not bool(torch.isfinite(delta).all().item()):
                    raise RuntimeError("Non-finite fallback/direct output difference")
                record["fallback_vs_direct_native"] = {
                    "exact": record["fallback_output"] == record["direct_native_output"],
                    "finite": bool(torch.isfinite(delta).all().item()),
                    "max_abs": float(delta.abs().max().item()),
                    "mae": float(delta.abs().mean().item()),
                    "rmse": float(delta.square().mean().sqrt().item()),
                    "value_scope": "Native floating VAE output, before image postprocessing; not uint8 RGB",
                }
                record["status"] = "CAPTURED"
                return result
            except Exception as exc:
                record["status"] = "FAILED"
                record["error"] = {"type": type(exc).__name__, "message": str(exc)}
                raise
            finally:
                record["ended_unix_s"] = time.time()
                text = json.dumps(record, sort_keys=True, allow_nan=False)
                (directory / f"{capture_id}.json").write_text(text + "\n")
                with (directory / "rank0-b1-captures.jsonl").open("a") as stream:
                    stream.write(text + "\n")
                print("OMNI_VAE_B1_DIAGNOSTIC_CAPTURE " + text, file=sys.stderr, flush=True)

        cls._batch_parallel_decode = decode

    class _Loader(importlib.abc.Loader):
        def __init__(self, delegate):
            self.delegate = delegate

        def create_module(self, spec):
            create = getattr(self.delegate, "create_module", None)
            return create(spec) if create else None

        def exec_module(self, module):
            self.delegate.exec_module(module)
            (_install_vae if module.__name__ == _VAE else _install_pipeline)(module)
            receipt = {
                "pid": os.getpid(),
                "module": module.__name__,
                "module_file": module.__file__,
                "module_sha256": _hash(Path(module.__file__)),
                "hook_sha256": _hash(Path(__file__)),
            }
            if module.__name__ == _VAE:
                native_file = Path(sys.modules[module.AutoencoderKLFlux2.__module__].__file__)
                receipt.update(
                    native_class_module=module.AutoencoderKLFlux2.__module__,
                    native_class_file=str(native_file),
                    native_class_sha256=_hash(native_file),
                )
            print("OMNI_VAE_B1_DIAGNOSTIC_HOOK " + json.dumps(receipt, sort_keys=True), file=sys.stderr, flush=True)

        def __getattr__(self, name):
            return getattr(self.delegate, name)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in (_VAE, _PIPELINE):
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _Loader(spec.loader)
            return spec

    sys.meta_path.insert(0, _Finder())
