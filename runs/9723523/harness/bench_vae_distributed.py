"""Benchmark the native DistributedAutoencoderKLFlux2 batch mode, including communication."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch
    import torch.distributed as dist
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_flux2 import DistributedAutoencoderKLFlux2
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    output = Path(args.output)
    torch.cuda.set_device(local_rank)
    init_distributed_environment(world_size=world, rank=rank, local_rank=local_rank, backend="nccl")
    initialize_model_parallel(data_parallel_size=1, sequence_parallel_size=world, allgather_degree=world, backend="nccl")
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        vae = DistributedAutoencoderKLFlux2.from_pretrained(
            args.model, subfolder="vae", local_files_only=True
        ).eval().to("cuda")
    finally:
        torch.set_default_dtype(previous_dtype)
    vae.disable_tiling()
    vae.disable_slicing()
    metadata = {
        "scope": "real VAE weights, native executor, synthetic latents; not end-to-end",
        "rank": rank,
        "world_size": world,
        "gpu": torch.cuda.get_device_name(),
        "gpu_memory_bytes": torch.cuda.get_device_properties(local_rank).total_memory,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "vllm", "diffusers", "transformers")},
        "cuda_build": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "vae_dtype": str(vae.dtype),
        "parameter_dtypes": sorted({str(p.dtype) for p in vae.parameters()}),
        "buffer_dtypes": sorted({str(b.dtype) for b in vae.buffers()}),
        "default_dtype": str(torch.get_default_dtype()),
        "autocast": torch.is_autocast_enabled(),
        "warmups": 2,
        "repeats": 8,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / f"metadata-rank{rank}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    checks = []
    generator = torch.Generator(device="cuda").manual_seed(8137)
    latents = torch.randn((5, 32, 128, 128), generator=generator, device="cuda", dtype=vae.dtype)

    def configure(arm):
        vae.set_parallel_size(world if arm == "batch" else 1, mode="batch" if arm == "batch" else "tile")

    def run_round(batch, arm, label):
        configure(arm)
        for _ in range(2):
            value = vae.decode(latents[:batch], return_dict=False)[0]
            del value
        torch.cuda.synchronize()
        samples = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(8):
            dist.barrier()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            value = vae.decode(latents[:batch], return_dict=False)[0]
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - begin) * 1000
            assert list(value.shape) == [batch, 3, 1024, 1024]
            del value
            all_times = [torch.empty(1, device="cuda", dtype=torch.float64) for _ in range(world)]
            dist.all_gather(all_times, torch.tensor([elapsed], device="cuda", dtype=torch.float64))
            samples.append([float(t.item()) for t in all_times])
        row = {
            "batch": batch,
            "arm": arm,
            "label": label,
            "per_rank_wall_ms": samples,
            "mean_slowest_rank_ms": statistics.mean(max(s) for s in samples),
            "std_slowest_rank_ms": statistics.stdev(max(s) for s in samples),
            "peak_allocated_bytes_local": torch.cuda.max_memory_allocated(),
        }
        rows.append(row)
        (output / f"timing-rank{rank}.json").write_text(json.dumps(rows, indent=2) + "\n")
        if rank == 0:
            print("ROUND", json.dumps(row), flush=True)

    with torch.inference_mode():
        for batch in [2, 4, 5, 1]:
            for arm, label in [("base", "base-a"), ("batch", "batch-a"), ("base", "base-b"), ("batch", "batch-b")]:
                run_round(batch, arm, label)
            # Same input on all ranks, preserved ordering and complete returned batch.
            configure("base")
            expected = vae.decode(latents[:batch], return_dict=False)[0]
            configure("batch")
            actual = vae.decode(latents[:batch], return_dict=True).sample
            diff = actual.float() - expected.float()
            check = {
                "batch": batch,
                "max_abs": diff.abs().max().item(),
                "mae": diff.abs().mean().item(),
                "rmse": diff.square().mean().sqrt().item(),
                "finite": bool(torch.isfinite(actual).all()),
                "dtype_matches": actual.dtype == expected.dtype,
                "shape_matches": actual.shape == expected.shape,
            }
            checks.append(check)
            assert check["finite"] and check["dtype_matches"] and check["shape_matches"], check
            del expected, actual, diff
        (output / f"correctness-rank{rank}.json").write_text(json.dumps(checks, indent=2) + "\n")
    del vae, latents
    torch.cuda.synchronize()
    dist.barrier()
    destroy_model_parallel()
    destroy_distributed_environment()
    (output / f"done-rank{rank}.json").write_text(json.dumps({"rank": rank, "cleanup_complete": True}) + "\n")
    print("RANK_COMPLETE", rank, flush=True)


if __name__ == "__main__":
    main()
