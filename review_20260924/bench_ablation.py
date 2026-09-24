# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pinned-source KV AllGather ablation, not a full-model benchmark.

Production class/method definitions are loaded unchanged via AST to avoid engine
initialization. Baseline and head use separate namespaces. Experimental arms
only override the head's gather helper. All collectives are real NCCL calls.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import gzip
import hashlib
import json
import os
import platform
import statistics
import sys
import time
import types
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

BASE_SHA = "3d43571b94f1683412023c45bd886f9ce30f76bd"
HEAD_SHA = "a2cae127c2933f0e754d38d3615da58660c952f1"
ARMS = ("baseline", "layout_only", "packing_only", "head", "head_contiguous")


def load_source(root, label):
    module = types.ModuleType(f"source_bound_kv_{label}")
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
        exec(compile(ast.fix_missing_locations(tree), f"{label}/{rel}", "exec"), ns)
    rel = "vllm_omni/diffusion/distributed/group_coordinator.py"
    source = (root / rel).read_text()
    hashes[rel] = hashlib.sha256(source.encode()).hexdigest()
    klass = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "GroupCoordinator")
    method = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "all_gather")
    tree = ast.parse("from __future__ import annotations\n")
    tree.body.append(method)
    exec(compile(ast.fix_missing_locations(tree), f"{label}/{rel}", "exec"), ns)
    return ns, hashes


def make_strategies(base_ns, head_ns, group_factory):
    candidate = head_ns["AllGatherKVParallelAttention"]
    assert hasattr(candidate, "_gather_kv")

    class LayoutOnly(candidate):
        def _gather_kv(self, key, value):
            def gather(tensor):
                packed = tensor.transpose(0, 1).contiguous()
                return self._sp_group.all_gather(packed, dim=0, group=self._allgather_group).transpose(0, 1)

            return gather(key), gather(value)

    class PackingOnly(candidate):
        def _gather_kv(self, key, value):
            packed = torch.stack((key, value), dim=2)
            gathered = self._sp_group.all_gather(packed, dim=1, group=self._allgather_group)
            return gathered.unbind(dim=2)

    class HeadContiguous(candidate):
        def _gather_kv(self, key, value):
            key, value = super()._gather_kv(key, value)
            return key.contiguous(), value.contiguous()

    return {
        "baseline": base_ns["AllGatherKVParallelAttention"](group_factory(base_ns)),
        "layout_only": LayoutOnly(group_factory(head_ns)),
        "packing_only": PackingOnly(group_factory(head_ns)),
        "head": candidate(group_factory(head_ns)),
        "head_contiguous": HeadContiguous(group_factory(head_ns)),
    }


def nccl_group(ns):
    class Group:
        all_gather = ns["all_gather"]

        def __init__(self):
            self.device_group = self.allgather_group = dist.group.WORLD
            self.allgather_world_size = dist.get_world_size()
            self.allgather_rank = dist.get_rank()

    return Group()


def attention(q, k, v):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=0.0, is_causal=False
        )


def full_attention(pre_fn):
    return attention(*pre_fn()[:3])


def max_across_ranks(values, dtype=torch.float64):
    tensor = torch.tensor(values, device="cuda", dtype=dtype)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.tolist()


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
    return max_across_ranks([start.elapsed_time(end) / iterations, (time.perf_counter() - wall) * 1000 / iterations])


def memory(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    initial = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    sizes = [torch.cuda.max_memory_allocated() - initial, torch.cuda.memory_allocated() - initial]
    del result
    return max_across_ranks(sizes, dtype=torch.int64)


def stats(values):
    return {
        "samples": values,
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def summarize(samples):
    return {"gpu_ms": stats([x[0] for x in samples]), "wall_ms": stats([x[1] for x in samples])}


def order_for_round(round_index, case_index=0):
    offset = (round_index + case_index) % len(ARMS)
    order = list(ARMS[offset:] + ARMS[:offset])
    return order[::-1] if round_index % 2 else order


def measure_arms(functions, args, case_index):
    samples = {name: [] for name in ARMS}
    orders = []
    for repeat in range(args.repeats):
        order = order_for_round(repeat, case_index)
        orders.append(order)
        for name in order:
            samples[name].append(measure(functions[name], args.iterations, args.warmup))
    return {"round_order": orders, "arms": {name: summarize(samples[name]) for name in ARMS}}


def make_inputs(case, world, rank, device="cuda"):
    b, s, h, d = case["batch"], case["seq"] // world, 32, 128
    gen = torch.Generator(device=device).manual_seed(8137 + rank)
    if case["layout"] == "projection_view":
        projection = torch.randn(b, s, h * 3, d, device=device, dtype=torch.bfloat16, generator=gen)
        return projection.split(h, dim=2)
    return tuple(torch.randn(b, s, h, d, device=device, dtype=torch.bfloat16, generator=gen) for _ in range(3))


def gather_diagnostic(strategy, fn):
    group = strategy._sp_group
    original = group.all_gather
    calls = []

    def counted(tensor, *args, **kwargs):
        calls.append(
            {
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "dim": kwargs.get("dim", args[0] if args else 0),
                "payload_bytes_per_rank": tensor.numel() * tensor.element_size(),
            }
        )
        return original(tensor, *args, **kwargs)

    group.all_gather = counted
    try:
        result = fn()
        output = {"q": list(result[0].stride()), "k": list(result[1].stride()), "v": list(result[2].stride())}
        del result
    finally:
        group.all_gather = original
    return {"all_gather_calls": calls, "count": len(calls), "output_strides": output}


def validate_eager(functions):
    baseline = functions["baseline"]()
    expected = attention(*baseline[:3])
    errors = {}
    for name in ARMS:
        actual = functions[name]()
        for left, right in zip(baseline[:3], actual[:3]):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        del left, right
        output = attention(*actual[:3])
        torch.testing.assert_close(output, expected, rtol=0.005, atol=0.005)
        errors[name] = max_across_ranks([(output.float() - expected.float()).abs().max().item()])[0]
        del actual, output
    del baseline, expected
    torch.cuda.synchronize()
    return {"qkv_exact": True, "attention_max_abs_error": errors, "attention_tolerance": {"rtol": 0.005, "atol": 0.005}}


def capture_graphs(functions, inputs, args, case_index):
    captures = {}
    try:
        # Separate pools; every arm remains alive through the interleaved timing.
        for name in order_for_round(0, case_index):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(args.warmup):
                    value = functions[name]()
                    del value
            stream.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            pool = torch.cuda.graph_pool_handle()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                output = functions[name]()
            torch.cuda.synchronize()
            captures[name] = (graph, output, stream, pool)

        saved = [tensor.clone() for tensor in inputs]
        for tensor in inputs:
            tensor.mul_(0.9)
        expected = functions["baseline"]()
        errors = {}
        for name in ARMS:
            graph, output, _, _ = captures[name]
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, expected, rtol=0.005, atol=0.005)
            errors[name] = max_across_ranks([(output.float() - expected.float()).abs().max().item()])[0]
        for tensor, original in zip(inputs, saved):
            tensor.copy_(original)
        del tensor, original, saved, expected, output, graph
        torch.cuda.synchronize()
        measured = measure_arms({name: captures[name][0].replay for name in ARMS}, args, case_index)
        measured["changed_input_max_abs_error"] = errors
        measured["capture_policy"] = "separate pools, all five captures alive, interleaved round order"
        return measured
    finally:
        torch.cuda.synchronize()
        for entry in captures.values():
            entry[0].reset()
        captures.clear()
        torch.cuda.synchronize()


def profile_arms(functions, output_dir, rank):
    """Untimed rank-0 profiler capture. Every rank executes the same collectives."""
    result = {"status": "PASS", "profiled_rank": 0, "iterations_per_arm": 3, "arms": {}}
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    for name in ARMS:
        for _ in range(3):
            value = functions[name]()
            del value
        torch.cuda.synchronize()
        dist.barrier()
        profiler = None
        init_error = None
        if rank == 0:
            try:
                profiler = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=False,
                    with_stack=False,
                )
                profiler.__enter__()
            except Exception as exc:
                init_error = f"{type(exc).__name__}: {exc}"
                profiler = None
        # Keep identical collective order even when profiler startup is unavailable.
        for _ in range(3):
            with torch.profiler.record_function(f"kv_ablation/{name}/pre_attention"):
                value = functions[name]()
                del value
        torch.cuda.synchronize()
        if rank == 0:
            if init_error is not None:
                result["status"] = "ERROR"
                result["arms"][name] = {"status": "ERROR", "error": init_error}
            else:
                try:
                    profiler.__exit__(None, None, None)
                    events = []
                    cuda_event_count = 0
                    for event in profiler.key_averages():
                        is_cuda = "CUDA" in str(event.device_type)
                        cuda_event_count += event.count if is_cuda else 0
                        events.append(
                            {
                                "name": event.key,
                                "count": event.count,
                                "device_type": str(event.device_type),
                                "self_cpu_time_us": event.self_cpu_time_total,
                                "cpu_time_us": event.cpu_time_total,
                                "self_device_time_us": getattr(event, "self_device_time_total", 0),
                                "device_time_us": getattr(event, "device_time_total", 0),
                            }
                        )
                    trace_path = output_dir / f"{name}.json"
                    profiler.export_chrome_trace(str(trace_path))
                    with trace_path.open("rb") as source, gzip.open(str(trace_path) + ".gz", "wb") as target:
                        target.write(source.read())
                    trace_path.unlink()
                    arm_status = "PASS" if cuda_event_count else "CPU_ONLY"
                    if arm_status != "PASS":
                        result["status"] = "INCOMPLETE"
                    result["arms"][name] = {
                        "status": arm_status,
                        "cuda_kernel_event_count": cuda_event_count,
                        "events": events,
                    }
                except Exception as exc:
                    result["status"] = "ERROR"
                    result["arms"][name] = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
        dist.barrier()
    if rank == 0:
        (output_dir / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def cpu_check(base_ns, head_ns):
    """Check rank/batch/layout ordering without CUDA; no performance claim."""
    count = 0
    for world in (2, 4):
        for batch in (1, 2):
            for layout in ("contiguous", "projection_view"):
                case = {"batch": batch, "seq": 16, "layout": layout}
                inputs = [make_inputs(case, world, rank, device="cpu") for rank in range(world)]
                for rank in range(world):

                    class Group:
                        allgather_group = None
                        allgather_world_size = world
                        allgather_rank = rank

                        def __init__(self):
                            self.call_index = 0
                            self.arm = None

                        def all_gather(self, tensor, dim=0, **kwargs):
                            i = self.call_index
                            self.call_index += 1
                            if self.arm == "baseline":
                                pieces = [entry[i + 1] for entry in inputs]
                            elif self.arm == "layout_only":
                                pieces = [entry[i + 1].transpose(0, 1).contiguous() for entry in inputs]
                            elif self.arm == "packing_only":
                                pieces = [torch.stack(entry[1:], dim=2) for entry in inputs]
                            else:
                                pieces = [
                                    torch.stack([x.transpose(0, 1) for x in entry[1:]], dim=2) for entry in inputs
                                ]
                            torch.testing.assert_close(tensor, pieces[rank], rtol=0, atol=0)
                            return torch.cat(pieces, dim=dim)

                    strategies = make_strategies(base_ns, head_ns, lambda _: Group())
                    for name, strategy in strategies.items():
                        strategy._sp_group.arm = name
                        actual = strategy.pre_attention(*inputs[rank], None)
                        expected = (
                            inputs[rank][0],
                            torch.cat([x[1] for x in inputs], dim=1),
                            torch.cat([x[2] for x in inputs], dim=1),
                        )
                        for left, right in zip(actual[:3], expected):
                            torch.testing.assert_close(left, right, rtol=0, atol=0)
                        count += 1
    print(json.dumps({"cpu_layout_check": "PASS", "arm_rank_layout_cases": count}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--head-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for root, expected in ((args.base_root, BASE_SHA), (args.head_root, HEAD_SHA)):
        assert (root / "COMMIT").read_text().strip() == expected, "Source snapshot COMMIT does not match pinned SHA"
    base_ns, base_hashes = load_source(args.base_root, "base")
    head_ns, head_hashes = load_source(args.head_root, "head")
    if args.check:
        cpu_check(base_ns, head_ns)
        return
    assert args.output is not None, "--output is required for GPU runs"
    assert args.iterations > 0 and args.repeats > 0 and args.warmup > 0
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.device("cuda", int(os.environ["LOCAL_RANK"])))
    rank, world = dist.get_rank(), dist.get_world_size()
    strategies = make_strategies(base_ns, head_ns, nccl_group)
    args.output.mkdir(parents=True, exist_ok=True)
    meta = {
        "base_sha": BASE_SHA,
        "head_sha": HEAD_SHA,
        "source_sha256": {"base": base_hashes, "head": head_hashes},
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
        "world_size": world,
        "gpu": torch.cuda.get_device_name(),
        "gpu_memory_bytes": torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory,
        "attention_backend": "torch SDPA FLASH_ATTENTION (forced)",
        "scope": "source-bound pre_attention plus local attention; no full engine or model",
        "dtype": "bfloat16",
        "query_heads": 32,
        "kv_heads": 32,
        "head_dim": 128,
        "causal": False,
        "warmup_per_sample": args.warmup,
        "iterations_per_sample": args.iterations,
        "rounds": args.repeats,
        "sample_definition": "mean per call over iterations, then maximum across ranks; spread across round means",
        "memory_definition": "eager pre_attention peak allocated minus entry allocation; validation outputs freed",
        "arms": {
            "baseline": "unchanged base class: separate K/V gathers, batch-first",
            "layout_only": "two separate gathers, each sequence-first and contiguous; transpose output views",
            "packing_only": "batch-first stack K/V, one gather dim=1, unbind output views",
            "head": "unchanged head class: sequence-first packed single gather",
            "head_contiguous": "head helper, then materialize contiguous K and V outputs",
        },
    }
    if rank == 0:
        if (args.output / "results.jsonl").exists():
            raise FileExistsError("Use a fresh output directory for each experiment")
        (args.output / "metadata.json").write_text(json.dumps(meta, indent=2))
    cases = [
        {"batch": 2, "seq": 4096, "layout": "contiguous"},
        {"batch": 2, "seq": 4096, "layout": "projection_view"},
        {"batch": 2, "seq": 16384, "layout": "contiguous"},
        {"batch": 1, "seq": 4096, "layout": "contiguous"},
    ]
    if args.all_cases:
        cases = [
            {"batch": b, "seq": s, "layout": layout}
            for b in (1, 2)
            for s in (1024, 4096, 16384)
            for layout in ("contiguous", "projection_view")
        ]
    with torch.inference_mode():
        for index, case in enumerate(cases):
            assert case["seq"] % world == 0
            q, k, v = make_inputs(case, world, rank)
            pre = {name: partial(strategies[name].pre_attention, q, k, v, None) for name in ARMS}
            full = {name: partial(full_attention, pre[name]) for name in ARMS}
            row = {"case": case, "validation": validate_eager(pre)}
            row["gather_diagnostic"] = {name: gather_diagnostic(strategies[name], pre[name]) for name in ARMS}
            for name, expected_calls in zip(ARMS, (2, 2, 1, 1, 1)):
                assert row["gather_diagnostic"][name]["count"] == expected_calls
            row["pre_attention_memory"] = {name: memory(pre[name]) for name in ARMS}
            row["eager_pre_attention"] = measure_arms(pre, args, index)
            row["eager_pre_plus_attention"] = measure_arms(full, args, index)
            row["graph_pre_plus_attention"] = capture_graphs(full, (q, k, v), args, index)
            if args.profile and case == {"batch": 2, "seq": 4096, "layout": "contiguous"}:
                profile = profile_arms(pre, args.output / "profile", rank)
                if rank == 0:
                    row["profile_status"] = profile["status"]
            if rank == 0:
                with (args.output / "results.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(f"CASE {index + 1}/{len(cases)} {case} PASS", flush=True)
            del q, k, v, pre, full, row
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    _, base_after = load_source(args.base_root, "base_after")
    _, head_after = load_source(args.head_root, "head_after")
    assert base_hashes == base_after and head_hashes == head_after, "Source changed during experiment"
    dist.barrier()
    torch.cuda.synchronize()
    dist.destroy_process_group()
    if rank == 0:
        (args.output / "validation.json").write_text(
            json.dumps(
                {"status": "PASS", "cases": len(cases), "source_unchanged": True, "cleanup_complete": True}, indent=2
            )
        )
        print("EXPERIMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
