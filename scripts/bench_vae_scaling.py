"""Real-weight VAE batch-scaling probe; not a full-model benchmark."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    print("IMPORT_START", flush=True)
    import torch
    from diffusers import AutoencoderKLFlux2
    from importlib.metadata import version

    assert torch.cuda.is_available()
    print("IMPORT_DONE", flush=True)
    # Match native Omni's default-dtype context and Klein's from_pretrained call.
    torch.set_default_dtype(torch.bfloat16)
    vae = AutoencoderKLFlux2.from_pretrained(args.model, subfolder="vae", local_files_only=True).eval().to("cuda")
    dtype = next(vae.parameters()).dtype
    vae.disable_tiling()
    vae.disable_slicing()
    model_files = sorted((Path(args.model) / "vae").glob("*.safetensors"))
    metadata = {
        "scope": "component scaling only; seeded synthetic latents, real pretrained VAE",
        "model": "black-forest-labs/FLUX.2-klein-4B",
        "model_revision": "e7b7dc27f91deacad38e78976d1f2b499d76a294",
        "vae_type": type(vae).__name__,
        "vae_dtype": str(dtype),
        "torch": version("torch"),
        "diffusers": version("diffusers"),
        "cuda_build": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "tiling": vae.use_tiling,
        "slicing": vae.use_slicing,
        "compile": False,
        "warmups_per_round": 2,
        "timed_repeats_per_round": 8,
        "seed": 8137,
        "latent_shape": [4, 32, 128, 128],
        "model_sha256": {p.name: digest(p) for p in model_files},
        "config_sha256": digest(Path(args.model) / "vae/config.json"),
        "harness_sha256": digest(Path(__file__)),
        "pjm_job_id": os.environ.get("PJM_JOBID"),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("MODEL_READY", json.dumps(metadata), flush=True)
    generator = torch.Generator(device="cuda").manual_seed(8137)
    latents = torch.randn((4, 32, 128, 128), device="cuda", dtype=dtype, generator=generator)
    results = []

    def decode(value):
        return vae.decode(value, return_dict=False)[0]

    def round_(batch, label):
        for _ in range(2):
            value = decode(latents[:batch])
            del value
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(8):
            torch.cuda.synchronize()
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start = time.perf_counter()
            begin.record()
            value = decode(latents[:batch])
            end.record()
            end.synchronize()
            samples.append({"wall_ms": (time.perf_counter() - start) * 1000, "gpu_ms": begin.elapsed_time(end)})
            assert list(value.shape) == [batch, 3, 1024, 1024], value.shape
            del value
        row = {
            "label": label,
            "batch": batch,
            "samples": samples,
            "mean_wall_ms": statistics.mean(x["wall_ms"] for x in samples),
            "std_wall_ms": statistics.stdev(x["wall_ms"] for x in samples),
            "mean_gpu_ms": statistics.mean(x["gpu_ms"] for x in samples),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        results.append(row)
        (output / "scaling.json").write_text(json.dumps(results, indent=2) + "\n")
        print("ROUND", json.dumps(row), flush=True)

    with torch.inference_mode():
        for label, batch in [("b1-a", 1), ("b2-a", 2), ("b4", 4), ("b2-b", 2), ("b1-b", 1)]:
            round_(batch, label)

        # Batch decomposition can change kernel choices: measure, don't assume equality.
        quality = []
        for batch in [2, 4]:
            full = decode(latents[:batch]).float()
            individual = torch.cat([decode(latents[i : i + 1]).float() for i in range(batch)])
            diff = full - individual
            quality.append({
                "batch": batch,
                "max_abs": diff.abs().max().item(),
                "mae": diff.abs().mean().item(),
                "rmse": diff.square().mean().sqrt().item(),
                "relative_l2": (diff.norm() / full.norm().clamp_min(1e-12)).item(),
                "cosine": torch.nn.functional.cosine_similarity(full.flatten(), individual.flatten(), dim=0).item(),
                "finite": bool(torch.isfinite(full).all() and torch.isfinite(individual).all()),
            })
            del full, individual, diff
        (output / "batch_decomposition.json").write_text(json.dumps(quality, indent=2) + "\n")
        print("QUALITY", json.dumps(quality), flush=True)

        # Diagnostic-only kernel evidence, after all scored rounds.
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
        ) as prof:
            with torch.profiler.record_function("vae_decode_b2"):
                value = decode(latents[:2])
                torch.cuda.synchronize()
        prof.export_chrome_trace(str(output / "trace-b2.json"))
        (output / "operators.txt").write_text(prof.key_averages().table(sort_by="self_device_time_total", row_limit=35))
        del value
    del vae, latents
    torch.cuda.synchronize()
    print("PROBE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
