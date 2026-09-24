# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Source-bound AllGather-KV experiment; PyTorch only, not an engine benchmark.

The AST loader executes the checkout's unchanged class/method definitions while
avoiding engine initialization and optional serving dependencies. Source hashes
are persisted. It does not emulate the collectives: all calls use real NCCL.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import os
import statistics
import sys
import time
import types
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def load_source(root):
    module = types.ModuleType("source_bound_kv")
    sys.modules[module.__name__] = module
    ns = module.__dict__
    ns.update(torch=torch, dataclass=dataclasses.dataclass, field=dataclasses.field, replace=dataclasses.replace)
    sources = [
        (
            "vllm_omni/diffusion/attention/backends/abstract.py",
            {"QueryRange", "VideoTokenSpan", "VideoTokenLayout", "PackedPaddingMetadata", "AttentionMetadata"},
        ),
        ("vllm_omni/diffusion/attention/parallel/base.py", {"ParallelAttentionContext"}),
        ("vllm_omni/diffusion/attention/parallel/allgather_kv.py", {"_AllGatherKVCtx", "AllGatherKVParallelAttention"}),
    ]
    hashes = {}
    for rel, names in sources:
        source = (root / rel).read_text()
        hashes[rel] = hashlib.sha256(source.encode()).hexdigest()
        nodes = [node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name in names]
        assert len(nodes) == len(names), (rel, names)
        tree = ast.parse("from __future__ import annotations\n")
        tree.body.extend(nodes)
        exec(compile(ast.fix_missing_locations(tree), str(root / rel), "exec"), ns)
    rel = "vllm_omni/diffusion/distributed/group_coordinator.py"
    source = (root / rel).read_text()
    hashes[rel] = hashlib.sha256(source.encode()).hexdigest()
    klass = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "GroupCoordinator")
    method = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "all_gather")
    tree = ast.parse("from __future__ import annotations\n")
    tree.body.append(method)
    exec(compile(ast.fix_missing_locations(tree), str(root / rel), "exec"), ns)
    return ns, hashes


def make_strategies(ns):
    class Group:
        all_gather = ns["all_gather"]

        def __init__(self):
            self.device_group = self.allgather_group = dist.group.WORLD
            self.allgather_world_size = dist.get_world_size()
            self.allgather_rank = dist.get_rank()

    candidate = ns["AllGatherKVParallelAttention"]
    assert hasattr(candidate, "_gather_kv"), "Expected the candidate's _gather_kv method"

    class Baseline(candidate):
        def _gather_kv(self, key, value):
            return (
                self._sp_group.all_gather(key, dim=1, group=self._allgather_group),
                self._sp_group.all_gather(value, dim=1, group=self._allgather_group),
            )

    class Materialized(candidate):
        def _gather_kv(self, key, value):
            k, v = super()._gather_kv(key, value)
            return k.contiguous(), v.contiguous()

    return {
        "baseline": Baseline(Group()),
        "coalesced": candidate(Group()),
        "coalesced_contiguous": Materialized(Group()),
    }


def attention(q, k, v):
    # Force the actual CUDA Flash SDPA kernel, not a quadratic math fallback.
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=q.shape[2] != k.shape[2],
        )


def measure(fn, iterations, warmup):
    for _ in range(warmup):
        result = fn()
        del result
    torch.cuda.synchronize()
    dist.barrier()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter()
    start.record()
    for _ in range(iterations):
        result = fn()
        del result
    end.record()
    end.synchronize()
    values = torch.tensor(
        [start.elapsed_time(end) / iterations, (time.perf_counter() - wall) * 1000 / iterations], device="cuda"
    )
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return values.tolist()


def memory(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    initial = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    sizes = torch.tensor(
        [torch.cuda.max_memory_allocated() - initial, torch.cuda.memory_allocated() - initial],
        dtype=torch.int64,
        device="cuda",
    )
    del result
    dist.all_reduce(sizes, op=dist.ReduceOp.MAX)
    return sizes.tolist()


def make_inputs(case, world, rank):
    b, s, hq, hkv, d = case["batch"], case["seq"] // world, 32, case["kv_heads"], 128
    gen = torch.Generator(device="cuda").manual_seed(8137 + rank)
    if case["layout"] == "projection_view":
        projection = torch.randn(b, s, hq + 2 * hkv, d, device="cuda", dtype=torch.bfloat16, generator=gen)
        q, k, v = projection.split([hq, hkv, hkv], dim=2)
    else:
        q = torch.randn(b, s, hq, d, device="cuda", dtype=torch.bfloat16, generator=gen)
        k = torch.randn(b, s, hkv, d, device="cuda", dtype=torch.bfloat16, generator=gen)
        v = torch.randn(b, s, hkv, d, device="cuda", dtype=torch.bfloat16, generator=gen)
    # Inputs vary by rank and batch; no all-zero/all-equal correctness shortcuts.
    return q, k, v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    rank, world = dist.get_rank(), dist.get_world_size()
    ns, hashes = load_source(args.source_root)
    strategies = make_strategies(ns)
    args.output.mkdir(parents=True, exist_ok=True)
    meta = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "world_size": world,
        "gpu": torch.cuda.get_device_name(),
        "source_sha256": hashes,
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "attention_backend": "torch SDPA FLASH_ATTENTION (forced)",
        "scope": "source-bound pre_attention + local attention; no full engine/model",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    if rank == 0:
        (args.output / "metadata.json").write_text(json.dumps(meta, indent=2))
    cases = [
        dict(batch=b, seq=s, kv_heads=h, layout=layout, joint="none")
        for b in (1, 2)
        for s in (1024, 4096, 16384)
        for h in (4, 32)
        for layout in ("projection_view", "contiguous")
    ]
    cases += [dict(batch=1, seq=4096, kv_heads=4, layout="contiguous", joint=j) for j in ("front", "rear")]
    if args.quick:
        cases = cases[:2]
    rows = []
    with torch.inference_mode():
        for index, case in enumerate(cases):
            q, k, v = make_inputs(case, world, rank)
            metadata = None
            if case["joint"] != "none":
                gen = torch.Generator(device="cuda").manual_seed(907)
                metadata = ns["AttentionMetadata"](
                    joint_query=torch.randn(1, 16, 32, 128, device="cuda", dtype=q.dtype, generator=gen),
                    joint_key=torch.randn(1, 24, case["kv_heads"], 128, device="cuda", dtype=k.dtype, generator=gen),
                    joint_value=torch.randn(1, 24, case["kv_heads"], 128, device="cuda", dtype=v.dtype, generator=gen),
                    joint_strategy=case["joint"],
                )

            def pre(name):
                return strategies[name].pre_attention(q, k, v, metadata)

            def full(name):
                result = pre(name)
                return attention(*result[:3])

            baseline = pre("baseline")
            reference = attention(*baseline[:3])
            errors = {}
            for name in ("coalesced", "coalesced_contiguous"):
                candidate = pre(name)
                for x, y in zip(baseline[:3], candidate[:3]):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)
                del x, y
                if metadata is not None:
                    assert baseline[3].query_ranges == candidate[3].query_ranges
                actual = attention(*candidate[:3])
                torch.testing.assert_close(actual, reference, rtol=0.005, atol=0.005)
                error = torch.tensor([(actual.float() - reference.float()).abs().max()], device="cuda")
                dist.all_reduce(error, op=dist.ReduceOp.MAX)
                errors[name] = error.item()
                del candidate, actual, error
            del baseline, reference
            dist.barrier()
            row = {"case": case, "correctness": "PASS", "max_abs_error": errors, "timings": {}}
            # Rotate positions to balance repeat-level drift; all arms include packing/allocation.
            names = list(strategies)
            for mode, call in (("pre_attention", pre), ("pre_plus_attention", full)):
                samples = {name: [] for name in names}
                for repeat in range(args.repeats):
                    offset = repeat % len(names)
                    order = names[offset:] + names[:offset]
                    for name in order:
                        samples[name].append(measure(lambda: call(name), args.iterations, args.warmup))
                row["timings"][mode] = {
                    name: {
                        "gpu_ms_samples": [x[0] for x in samples[name]],
                        "wall_ms_samples": [x[1] for x in samples[name]],
                        "gpu_ms_median": statistics.median(x[0] for x in samples[name]),
                        "wall_ms_median": statistics.median(x[1] for x in samples[name]),
                        "memory_increment_bytes": memory(lambda: call(name)),
                    }
                    for name in names
                }
            rows.append(row)
            if rank == 0:
                with (args.output / "results.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                t = row["timings"]["pre_plus_attention"]
                ratio = t["baseline"]["gpu_ms_median"] / t["coalesced"]["gpu_ms_median"]
                print(f"CASE {index + 1}/{len(cases)} {case} PASS speedup={ratio:.4f}", flush=True)
    _, after = load_source(args.source_root)
    assert hashes == after, "Source changed during experiment"
    dist.barrier()
    torch.cuda.synchronize()
    dist.destroy_process_group()
    if rank == 0:
        (args.output / "validation.json").write_text(
            json.dumps(
                {"status": "PASS", "cases": len(rows), "source_unchanged": True, "cleanup_complete": True}, indent=2
            )
        )
        print("EXPERIMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
