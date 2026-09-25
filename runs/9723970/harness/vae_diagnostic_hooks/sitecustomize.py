"""Opt-in target selection for the frozen Klein pipeline's native profiler.

Enable only with launcher --diagnostic --diagnostic-hook-dir THIS_DIRECTORY
and OMNI_VAE_BATCH_DIAGNOSTIC=1. Imports no model or torch at interpreter startup.
"""

import os

if os.environ.get("OMNI_VAE_BATCH_DIAGNOSTIC") == "1":
    import hashlib
    import importlib.abc
    import importlib.machinery
    import json
    import sys
    from pathlib import Path

    _TARGET = "vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein"
    _TARGETS = [
        "text_encoder.forward",
        "transformer.forward",
        "vae.encode",
        "vae.decode",
    ]

    class _DiagnosticLoader(importlib.abc.Loader):
        def __init__(self, delegate):
            self.delegate = delegate

        def create_module(self, spec):
            create = getattr(self.delegate, "create_module", None)
            return create(spec) if create is not None else None

        def exec_module(self, module):
            self.delegate.exec_module(module)
            pipeline = module.Flux2KleinPipeline
            pipeline._PROFILER_TARGETS = list(_TARGETS)
            print(
                "OMNI_VAE_BATCH_DIAGNOSTIC_HOOK "
                + json.dumps(
                    {
                        "pid": os.getpid(),
                        "module": module.__name__,
                        "module_file": module.__file__,
                        "targets": pipeline._PROFILER_TARGETS,
                        "hook_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )

        def __getattr__(self, name):
            return getattr(self.delegate, name)

    class _DiagnosticFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != _TARGET:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _DiagnosticLoader(spec.loader)
            return spec

    sys.meta_path.insert(0, _DiagnosticFinder())
