"""Opt-in, diagnostic-only hooks for the full-model KV AllGather experiment.

Add this directory to PYTHONPATH and set KV_MODEL_DIAGNOSTIC_DIR in a separate
diagnostic run. Without that variable this module does not install any hooks.
No torch/vLLM module is imported here: the driver can choose its source checkout
before importing Omni. Spawned workers inherit the same opt-in environment.

Every record is appended and closed immediately. Timings include instrumentation
overhead and must not be used as scored latency. pre_attention counts are Python
invocations, including capture/warmup calls, not CUDA Graph replay kernel counts.
Encoder/VAE measurements belong to Omni's separate built-in pipeline profiler.
"""

import os

if os.environ.get("KV_MODEL_DIAGNOSTIC_DIR"):
    import contextvars
    import functools
    import hashlib
    import importlib.abc
    import importlib.machinery
    import itertools
    import json
    import socket
    import sys
    import time
    from pathlib import Path

    _TARGETS = {
        "vllm_omni.diffusion.attention.parallel.allgather_kv": ("AllGatherKVParallelAttention", "pre_attention"),
        "vllm_omni.diffusion.models.flux2.flux2_transformer": ("Flux2Transformer2DModel", "forward"),
        "vllm_omni.diffusion.models.flux2_klein.flux2_klein_transformer": ("Flux2Transformer2DModel", "forward"),
    }
    _ROOT = Path(os.environ["KV_MODEL_DIAGNOSTIC_DIR"]).expanduser()
    _HOST = socket.gethostname()
    _COUNTS = {}
    _SEEN = set()
    _WARNED = set()
    _ACTIVE_FORWARD = contextvars.ContextVar("kv_model_diagnostic_forward", default=None)

    def _write(event, **fields):
        """Best-effort observation must not replace a model result or exception."""
        try:
            pid = os.getpid()
            record = {
                "event": event,
                "diagnostic_only": True,
                "pid": pid,
                "ppid": os.getppid(),
                "host": _HOST,
                "time_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
                "arm": os.environ.get("KV_MODEL_ARM"),
                "run_id": os.environ.get("KV_MODEL_RUN_ID"),
                "local_rank_env": os.environ.get("LOCAL_RANK"),
            }
            torch = sys.modules.get("torch")
            dist = getattr(torch, "distributed", None)
            if dist is not None and dist.is_available() and dist.is_initialized():
                record.update(rank=dist.get_rank(), world_size=dist.get_world_size())
            record.update(fields)
            _ROOT.mkdir(parents=True, exist_ok=True)
            with (_ROOT / f"events-{_HOST}-{pid}.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception as exc:
            warning = (os.getpid(), type(exc).__name__)
            if warning not in _WARNED:
                _WARNED.add(warning)
                try:
                    sys.__stderr__.write(f"KV diagnostic record failed: {type(exc).__name__}\n")
                except Exception:
                    pass

    def _next_count(name):
        key = (os.getpid(), name)
        return next(_COUNTS.setdefault(key, itertools.count(1)))

    def _tensor_info(value, torch):
        if not isinstance(value, torch.Tensor):
            return None
        return {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "contiguous": value.is_contiguous(),
        }

    def _argument(args, kwargs, name, position):
        return kwargs[name] if name in kwargs else args[position] if len(args) > position else None

    def _cuda_tensor(value, torch):
        if isinstance(value, torch.Tensor):
            # FakeTensor/meta tracing must not initialize or synchronize CUDA.
            if hasattr(value, "fake_mode") or value.device.type != "cuda":
                return None
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                found = _cuda_tensor(item, torch)
                if found is not None:
                    return found
        elif isinstance(value, dict):
            for item in value.values():
                found = _cuda_tensor(item, torch)
                if found is not None:
                    return found
        return None

    def _capture_state(torch, tensor):
        if tensor is None:
            return None
        try:
            with torch.cuda.device(tensor.device):
                return torch.cuda.is_current_stream_capturing()
        except Exception as exc:
            _write("instrumentation_error", operation="capture_state", error_type=type(exc).__name__)
            return None

    def _wrap_pre_attention(original, module_name):
        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            count = _next_count(module_name + ".pre_attention")
            details = {}
            try:
                torch = sys.modules["torch"]
                query = _argument(args, kwargs, "query", 0)
                key = _argument(args, kwargs, "key", 1)
                value = _argument(args, kwargs, "value", 2)
                metadata = _argument(args, kwargs, "attn_metadata", 3)
                details = {
                    "inputs": {
                        name: _tensor_info(tensor, torch)
                        for name, tensor in (("query", query), ("key", key), ("value", value))
                    },
                    "joint": {
                        name: _tensor_info(getattr(metadata, name, None), torch)
                        for name in ("joint_query", "joint_key", "joint_value")
                    },
                    "joint_strategy": getattr(metadata, "joint_strategy", None),
                    "sp_size": getattr(self, "_sp_size", None),
                    "sp_rank": getattr(self, "_sp_rank", None),
                    "cuda_stream_capturing": _capture_state(torch, _cuda_tensor((query, key, value), torch)),
                }
            except Exception as exc:
                _write("instrumentation_error", operation="pre_attention_inputs", error_type=type(exc).__name__)
            try:
                result = original(self, *args, **kwargs)
            except BaseException as exc:
                _write(
                    "pre_attention",
                    module=module_name,
                    call_index=count,
                    forward=_ACTIVE_FORWARD.get(),
                    status="error",
                    error_type=type(exc).__name__,
                )
                raise
            try:
                details["outputs"] = {
                    name: _tensor_info(tensor, torch) for name, tensor in zip(("query", "key", "value"), result[:3])
                }
                signature = (os.getpid(), json.dumps(details, sort_keys=True))
                representative = signature not in _SEEN
                _SEEN.add(signature)
                _write(
                    "pre_attention",
                    module=module_name,
                    call_index=count,
                    forward=_ACTIVE_FORWARD.get(),
                    status="ok",
                    representative=details if representative else None,
                )
            except Exception as exc:
                _write(
                    "pre_attention",
                    module=module_name,
                    call_index=count,
                    forward=_ACTIVE_FORWARD.get(),
                    status="ok",
                    observation_error=type(exc).__name__,
                )
            return result

        return wrapped

    def _wrap_forward(original, module_name):
        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            count = _next_count(module_name + ".forward")
            forward_id = {"module": module_name, "call_index": count}
            timing = None
            details = {"timing_status": "no_cuda_input"}
            try:
                torch = sys.modules["torch"]
                tensor = _cuda_tensor((args, kwargs), torch)
                if tensor is not None:
                    details["input"] = _tensor_info(tensor, torch)
                    compiling = getattr(getattr(torch, "compiler", None), "is_compiling", lambda: False)()
                    capturing = None if compiling else _capture_state(torch, tensor)
                    details.update(compiling=compiling, cuda_stream_capturing=capturing)
                    if compiling or capturing is not False:
                        details["timing_status"] = "skipped_compile_or_capture"
                    else:
                        with torch.cuda.device(tensor.device):
                            torch.cuda.synchronize(tensor.device)
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            stream = torch.cuda.current_stream(tensor.device)
                            start.record(stream)
                        timing = (start, end, stream, tensor.device)
                        details["timing_status"] = "cuda_events"
            except Exception as exc:
                details.update(timing_status="instrumentation_error", timing_error_type=type(exc).__name__)
            context_token = _ACTIVE_FORWARD.set(forward_id)
            wall_start = time.perf_counter_ns()
            try:
                result = original(self, *args, **kwargs)
            except BaseException as exc:
                _write("transformer_forward", **forward_id, status="error", error_type=type(exc).__name__, **details)
                raise
            else:
                if timing is not None:
                    try:
                        start, end, stream, device = timing
                        with torch.cuda.device(device):
                            end.record(stream)
                            end.synchronize()
                            details["cuda_ms"] = start.elapsed_time(end)
                    except Exception as exc:
                        details.update(timing_status="instrumentation_error", timing_error_type=type(exc).__name__)
                details["wall_ms"] = (time.perf_counter_ns() - wall_start) / 1_000_000
                _write("transformer_forward", **forward_id, status="ok", **details)
                return result
            finally:
                _ACTIVE_FORWARD.reset(context_token)

        return wrapped

    def _patch(module):
        try:
            class_name, method_name = _TARGETS[module.__name__]
            cls = getattr(module, class_name)
            original = getattr(cls, method_name)
            if getattr(original, "_kv_model_diagnostic_wrapper", False):
                return
            wrapper = _wrap_pre_attention if method_name == "pre_attention" else _wrap_forward
            wrapped = wrapper(original, module.__name__)
            wrapped._kv_model_diagnostic_wrapper = True
            setattr(cls, method_name, wrapped)
            source = Path(module.__file__).resolve()
            _write(
                "module_patched",
                module=module.__name__,
                class_name=class_name,
                method=method_name,
                source_path=str(source),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            )
        except Exception as exc:
            _write(
                "instrumentation_error", operation="patch_module", module=module.__name__, error_type=type(exc).__name__
            )

    class _DiagnosticLoader(importlib.abc.Loader):
        def __init__(self, original):
            self.original = original

        def create_module(self, spec):
            create = getattr(self.original, "create_module", None)
            return create(spec) if create is not None else None

        def exec_module(self, module):
            self.original.exec_module(module)
            _patch(module)

        def __getattr__(self, name):
            return getattr(self.original, name)

    class _DiagnosticFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname not in _TARGETS:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _DiagnosticLoader(spec.loader)
            return spec

    sys.meta_path.insert(0, _DiagnosticFinder())
