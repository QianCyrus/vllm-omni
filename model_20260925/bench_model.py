#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Full FLUX.2-klein-4B generation through the actual Omni runtime.

Run once per frozen base/head checkout. Do not torchrun this driver: Omni starts
its own workers. Scored runs have no benchmark hooks or pipeline profiler.
--diagnostic enables the built-in pipeline profiler; an optional external
sitecustomize hook can record actual AllGather-KV calls and transformer CUDA
events in separate runs. No production source is patched by this driver.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
MODEL_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
SOURCE_FILES = (
    "vllm_omni/diffusion/attention/parallel/allgather_kv.py",
    "vllm_omni/diffusion/attention/parallel/factory.py",
    "vllm_omni/diffusion/attention/layer.py",
    "vllm_omni/diffusion/attention/backends/sdpa.py",
    "vllm_omni/diffusion/distributed/group_coordinator.py",
    "vllm_omni/diffusion/distributed/sp_plan.py",
    "vllm_omni/diffusion/models/flux2_klein/flux2_klein_transformer.py",
    "vllm_omni/diffusion/models/flux2_klein/pipeline_flux2_klein.py",
    "vllm_omni/diffusion/worker/diffusion_model_runner.py",
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(root):
    return {name: sha256(root / name) for name in SOURCE_FILES}


def verify_checkout(root, expected_sha):
    """Verify a git checkout or an exported snapshot with a COMMIT marker."""
    if (root / ".git").exists():
        actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        if actual != expected_sha:
            raise ValueError(f"Checkout is {actual}, expected {expected_sha}")
        subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--"], check=True)
    else:
        marker = root / "COMMIT"
        if not marker.is_file() or marker.read_text().strip() != expected_sha:
            raise ValueError("Exported source requires a COMMIT file matching --source-sha")


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed as a distribution"


def stats(values):
    return {
        "samples": values,
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def save_images(images, output_dir, label, width, height):
    records = []
    for index, image in enumerate(images):
        image = image.convert("RGB")
        if image.size != (width, height):
            raise AssertionError(f"Unexpected image size {image.size}, expected {(width, height)}")
        relative = f"images/{label}_{index}.png"
        path = output_dir / relative
        image.save(path, format="PNG")
        records.append(
            {
                "file": relative,
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "png_sha256": sha256(path),
            }
        )
    return records


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local complete snapshot of the official model")
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=8137)
    parser.add_argument(
        "--prompt",
        default="A small red sailboat on a calm blue lake, distant green hills, soft morning light, realistic photograph.",
    )
    parser.add_argument("--diagnostic", action="store_true", help="Instrumented run; exclude from scored latency")
    parser.add_argument("--init-timeout", type=int, default=900)
    parser.add_argument(
        "--check", action="store_true", help="Validate source/model config and arguments without imports or CUDA"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.source_root = args.source_root.resolve()
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    if args.world_size < 2 or args.batch < 1 or args.steps < 1 or args.repeats < 1 or args.warmup < 0:
        raise ValueError("Invalid world size, batch, steps, repeats, or warmup")
    if args.height % 16 or args.width % 16:
        raise ValueError("Use height and width divisible by 16")
    if ((args.height // 16) * (args.width // 16)) % args.world_size:
        raise ValueError("Image-token count must divide by world size for this mask-free benchmark")
    if len(args.source_sha) != 40 or any(c not in "0123456789abcdef" for c in args.source_sha):
        raise ValueError("--source-sha must be a full lowercase git SHA")
    if not args.diagnostic and os.environ.get("KV_MODEL_DIAGNOSTIC_DIR"):
        raise ValueError("Unset KV_MODEL_DIAGNOSTIC_DIR for scored runs")
    verify_checkout(args.source_root, args.source_sha)
    before = source_hashes(args.source_root)
    config_names = (
        "model_index.json",
        "transformer/config.json",
        "text_encoder/config.json",
        "vae/config.json",
        "scheduler/scheduler_config.json",
    )
    model_hashes = {name: sha256(args.model / name) for name in config_names}
    model_index = json.loads((args.model / "model_index.json").read_text())
    transformer_config = json.loads((args.model / "transformer/config.json").read_text())
    if model_index.get("_class_name") != "Flux2KleinPipeline":
        raise ValueError("The model snapshot must declare Flux2KleinPipeline")
    if args.check:
        print(
            json.dumps(
                {
                    "static_preflight": "PASS",
                    "source_sha": args.source_sha,
                    "source_files": len(before),
                    "model_config_files": len(model_hashes),
                    "model_class": model_index["_class_name"],
                }
            )
        )
        return
    if (args.output / "metadata.json").exists() or (args.output / "results.jsonl").exists():
        raise FileExistsError("Use a fresh output directory for each arm/run")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "images").mkdir(exist_ok=True)

    # Spawned workers must import this exact checkout, including when launched
    # from an installed vLLM/Omni environment. Existing optional diagnostic hook
    # directories remain available after the source root in PYTHONPATH.
    sys.path.insert(0, str(args.source_root))
    os.environ["PYTHONPATH"] = str(args.source_root) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    metadata = {
        "status": "STARTING",
        "source_sha": args.source_sha,
        "source_sha256": before,
        "driver_sha256": sha256(Path(__file__)),
        "model": MODEL_ID,
        "model_revision": args.model_revision,
        "model_config_sha256": model_hashes,
        "python": platform.python_version(),
        "world_size": args.world_size,
        "batch": args.batch,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "warmup_requests": args.warmup,
        "measured_requests": args.repeats,
        "prompt": args.prompt,
        "seed": args.seed,
        "generator_policy": "Fresh CPU torch.Generator with the same seed for every request; one generator draws the batch of outputs",
        "diagnostic": args.diagnostic,
        "diagnostic_hooks_enabled": bool(os.environ.get("KV_MODEL_DIAGNOSTIC_DIR")),
        "timing_scope": "Omni.generate wall time: prompt encoding, denoising, VAE decode and output transport; excludes startup and PNG writing",
        "memory_scope": "Worker-reported peak reserved MiB, not the previous operator allocated-increment metric",
        "attention_backend": "TORCH_SDPA",
        "transformer_config": {
            key: transformer_config.get(key)
            for key in (
                "num_layers",
                "num_single_layers",
                "num_attention_heads",
                "attention_head_dim",
                "in_channels",
                "guidance_embeds",
            )
        },
        "limitations": [
            "One model, one prompt and one seeded batch; pixel equivalence does not establish general image quality.",
            "Pipeline profiler adds synchronization; diagnostic timings are not scored.",
            "Klein denoises inline, so its built-in profiler has no diffuse stage; use separate transformer hooks for DiT attribution.",
        ],
    }
    write_json(args.output / "metadata.json", metadata)
    omni = None
    success = False
    rows = []
    import_start = time.perf_counter()
    try:
        import torch
        import vllm
        import vllm_omni

        if not Path(vllm_omni.__file__).resolve().is_relative_to(args.source_root):
            raise RuntimeError("Imported vllm_omni does not belong to --source-root")
        from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
        from vllm_omni.entrypoints.omni import Omni
        from vllm_omni.inputs.data import OmniDiffusionSamplingParams

        metadata["imports_seconds"] = time.perf_counter() - import_start
        metadata["versions"] = {
            name: package_version(name)
            for name in (
                "torch",
                "vllm",
                "vllm-omni",
                "diffusers",
                "transformers",
                "flash-attn",
                "Pillow",
                "huggingface-hub",
            )
        }
        metadata["vllm_runtime_version"] = getattr(vllm, "__version__", None)
        metadata["vllm_omni_runtime_version"] = getattr(vllm_omni, "__version__", None)
        metadata["cuda_build"] = torch.version.cuda
        metadata["nccl"] = torch.cuda.nccl.version()
        if torch.cuda.device_count() < args.world_size:
            raise RuntimeError("Fewer visible GPUs than --world-size")
        metadata["gpus"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_bytes": torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(args.world_size)
        ]
        engine_args = {
            "model": str(args.model),
            "model_class_name": "Flux2KleinPipeline",
            "num_gpus": args.world_size,
            "dtype": "bfloat16",
            "allgather_degree": args.world_size,
            "tensor_parallel_size": 1,
            "ulysses_degree": 1,
            "ring_degree": 1,
            "cfg_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "vae_patch_parallel_size": 1,
            "diffusion_attention_backend": "TORCH_SDPA",
            "enforce_eager": True,
            "cache_backend": "none",
            "enable_prompt_embed_cache": False,
            "enable_cpu_offload": False,
            "enable_layerwise_offload": False,
            "enable_distributed_layerwise_offload": False,
            "use_hsdp": False,
            "quantization": None,
            "enable_diffusion_pipeline_profiler": args.diagnostic,
            "init_timeout": args.init_timeout,
            "stage_init_timeout": args.init_timeout,
        }
        metadata["engine_args"] = {key: (MODEL_ID if key == "model" else value) for key, value in engine_args.items()}
        write_json(args.output / "metadata.json", metadata)
        start = time.perf_counter()
        omni = Omni(**engine_args)
        metadata["startup_seconds"] = time.perf_counter() - start
        metadata["status"] = "RUNNING"
        write_json(args.output / "metadata.json", metadata)

        for request_index in range(args.warmup + args.repeats):
            is_warmup = request_index < args.warmup
            sample_index = request_index if is_warmup else request_index - args.warmup
            label = f"{'warmup' if is_warmup else 'sample'}_{sample_index:02d}"
            params = OmniDiffusionSamplingParams(
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                num_outputs_per_prompt=args.batch,
                guidance_scale=1.0,
                max_sequence_length=512,
                seed=args.seed,
                generator=torch.Generator(device="cpu").manual_seed(args.seed),
                generator_device="cpu",
                output_type="pil",
            )
            prompt = {"prompt": args.prompt, "modalities": ["image"]}
            start = time.perf_counter()
            outputs = omni.generate(prompt, params, use_tqdm=False)
            elapsed = time.perf_counter() - start
            if not outputs or any(getattr(output, "error", None) for output in outputs):
                raise RuntimeError(f"Generation failed: {[getattr(output, 'error', None) for output in outputs]}")
            images = extract_images_from_outputs(outputs)
            if len(images) != args.batch:
                raise AssertionError(f"Expected {args.batch} images, received {len(images)}")
            row = {
                "request_index": request_index,
                "sample_index": sample_index,
                "warmup": is_warmup,
                "diagnostic": args.diagnostic,
                "wall_seconds": elapsed,
                "images": save_images(images, args.output, label, args.width, args.height),
                "worker_peak_reserved_mib": [float(getattr(output, "peak_memory_mb", 0.0)) for output in outputs],
                "stage_durations_raw": [dict(getattr(output, "stage_durations", {})) for output in outputs],
            }
            rows.append(row)
            with (args.output / "results.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(f"{label} wall={elapsed:.6f}s images={len(images)}", flush=True)
            del outputs, images, params
        measured = [row for row in rows if not row["warmup"]]
        first_hashes = [image["pixel_sha256"] for image in measured[0]["images"]]
        repeated_pixels_equal = all(
            [image["pixel_sha256"] for image in row["images"]] == first_hashes for row in measured
        )
        write_json(
            args.output / "summary.json",
            {
                "source_sha": args.source_sha,
                "diagnostic": args.diagnostic,
                "scored": not args.diagnostic,
                "wall_seconds": stats([row["wall_seconds"] for row in measured]),
                "same_seed_repeated_pixels_exact": repeated_pixels_equal,
                "pixel_sha256": first_hashes,
                "note": "Exact repeat equality is reported, not assumed; cross-arm comparison requires both output sets.",
            },
        )
        if source_hashes(args.source_root) != before:
            raise AssertionError("Source files changed during the run")
        verify_checkout(args.source_root, args.source_sha)
        success = True
    except BaseException as exc:
        write_json(
            args.output / "failure.json",
            {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
        )
        raise
    finally:
        cleanup_error = None
        if omni is not None:
            try:
                omni.close()
            except BaseException as exc:
                cleanup_error = f"{type(exc).__name__}: {exc}"
        metadata["status"] = "PASS" if success and cleanup_error is None else "FAIL"
        metadata["cleanup_complete"] = omni is not None and cleanup_error is None
        metadata["cleanup_error"] = cleanup_error
        write_json(args.output / "metadata.json", metadata)
        write_json(
            args.output / "validation.json",
            {
                "status": metadata["status"],
                "completed_requests": len(rows),
                "source_unchanged": source_hashes(args.source_root) == before,
                "cleanup_complete": metadata["cleanup_complete"],
            },
        )
        if cleanup_error is not None and success:
            raise RuntimeError(f"Cleanup failed: {cleanup_error}")
    print("EXPERIMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
