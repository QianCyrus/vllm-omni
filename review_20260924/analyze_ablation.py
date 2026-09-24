#!/usr/bin/env python3
"""Summarize saved KV ablations without modifying the input artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path

ARMS = ("baseline", "layout_only", "packing_only", "head", "head_contiguous")
STAGES = ("eager_pre_attention", "eager_pre_plus_attention", "graph_pre_plus_attention")
CONTRASTS = (
    ("baseline", "layout_only"),
    ("baseline", "packing_only"),
    ("layout_only", "head"),
    ("head", "head_contiguous"),
)
EXPECTED_CALLS = dict(zip(ARMS, (2, 2, 1, 1, 1)))
MIB = 1024 * 1024


def read_json(path):
    return json.loads(path.read_text())


def discover_runs(paths):
    found = set()
    for path in paths:
        if path.is_file() and path.name == "results.jsonl":
            found.add(path.parent.resolve())
        elif (path / "results.jsonl").exists():
            found.add(path.resolve())
        elif path.is_dir():
            found.update(p.parent.resolve() for p in path.rglob("results.jsonl"))
        else:
            raise FileNotFoundError(path)
    if not found:
        raise ValueError("No results.jsonl found under --runs")
    return sorted(found)


def trace_counts(path):
    with gzip.open(path, "rt") as stream:
        trace = json.load(stream)
    kernels, aten = Counter(), Counter()
    for event in trace.get("traceEvents", []):
        if event.get("ph") != "X":
            continue
        name, category = event.get("name", ""), event.get("cat", "").lower()
        if "kernel" in category and "nccl" in name.lower():
            kernels[name] += 1
        if name in ("aten::clone", "aten::copy_", "aten::contiguous"):
            aten[name] += 1
    return dict(kernels), dict(aten)


def summarize_profile(run_dir):
    path = run_dir / "profile" / "summary.json"
    if not path.exists():
        return {"status": "NOT_REQUESTED_OR_NO_ARTIFACT"}
    raw = read_json(path)
    result = {
        "status": raw.get("status", "UNKNOWN"),
        "profiled_rank": raw.get("profiled_rank"),
        "iterations_per_arm": raw.get("iterations_per_arm"),
        "note": "Counts cover three untimed pre_attention calls on rank 0. Nested events overlap; no durations are summed.",
        "arms": {},
    }
    for name, arm in raw.get("arms", {}).items():
        kernels, aten = Counter(), Counter()
        for event in arm.get("events", []):
            key = event.get("name", "")
            # CUDA annotations such as nccl:_all_gather_base are not kernels.
            if "CUDA" in event.get("device_type", "").upper() and "nccldevkernel" in key.lower():
                kernels[key] += event["count"]
            if key in ("aten::clone", "aten::copy_", "aten::contiguous"):
                aten[key] += event["count"]
        source = "profiler key_averages, ncclDevKernel names only (trace unavailable)"
        trace = path.parent / f"{name}.json.gz"
        if trace.exists():
            trace_kernels, trace_aten = trace_counts(trace)
            kernels = Counter(trace_kernels)
            if not aten:
                aten = Counter(trace_aten)
            source = "trace category kernel only (excludes CUDA annotations); operator counts from key_averages"
        result["arms"][name] = {
            "status": arm.get("status", "UNKNOWN"),
            "count_source": source,
            "nccl_kernel_count": sum(kernels.values()),
            "nccl_kernels": dict(kernels),
            "aten_counts": {key: aten.get(key, 0) for key in ("aten::clone", "aten::copy_", "aten::contiguous")},
        }
        if "error" in arm:
            result["arms"][name]["error"] = arm["error"]
    return result


def summarize_run(run_dir):
    meta = read_json(run_dir / "metadata.json")
    validation_path = run_dir / "validation.json"
    validation = read_json(validation_path) if validation_path.exists() else {"status": "INCOMPLETE"}
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines() if line.strip()]
    result = {
        "run": run_dir.name,
        "metadata": meta,
        "validation": validation,
        "cases": [],
        "profile": summarize_profile(run_dir),
    }
    csv_rows = []
    for row in rows:
        case = row["case"]
        item = {
            "case": case,
            "validation": row["validation"],
            "changed_input_max_abs_error": row["graph_pre_plus_attention"]["changed_input_max_abs_error"],
            "arms": {},
            "contrasts": {},
        }
        for name in ARMS:
            diagnostic = row["gather_diagnostic"][name]
            peak, retained = row["pre_attention_memory"][name]
            item["arms"][name] = {
                "all_gather_calls": diagnostic["count"],
                "expected_all_gather_calls": EXPECTED_CALLS[name],
                "call_count_matches": diagnostic["count"] == EXPECTED_CALLS[name],
                "input_bytes_per_rank": sum(c["payload_bytes_per_rank"] for c in diagnostic["all_gather_calls"]),
                "pre_attention_peak_allocated_increment_mib": peak / MIB,
                "pre_attention_retained_increment_mib": retained / MIB,
                "timings": {stage: row[stage]["arms"][name] for stage in STAGES},
            }
            for stage in STAGES:
                timing = row[stage]["arms"][name]
                flat = {
                    "run": run_dir.name,
                    "gpus": meta["world_size"],
                    **case,
                    "arm": name,
                    "stage": stage,
                    "all_gather_calls": diagnostic["count"],
                    "expected_all_gather_calls": EXPECTED_CALLS[name],
                    "pre_attention_peak_allocated_increment_mib": peak / MIB,
                    "pre_attention_retained_increment_mib": retained / MIB,
                }
                for clock in ("gpu_ms", "wall_ms"):
                    flat.update(
                        {f"{clock}_{key}": timing[clock][key] for key in ("mean", "median", "sample_std", "min", "max")}
                    )
                csv_rows.append(flat)
        for before, after in CONTRASTS:
            contrast = {}
            for stage in STAGES:
                old = row[stage]["arms"][before]["gpu_ms"]["mean"]
                new = row[stage]["arms"][after]["gpu_ms"]["mean"]
                contrast[stage] = {
                    "before_gpu_ms_mean": old,
                    "after_gpu_ms_mean": new,
                    "saved_gpu_ms": old - new,
                    "latency_reduction_percent": (old - new) / old * 100,
                }
            item["contrasts"][f"{before} -> {after}"] = contrast
        result["cases"].append(item)
    return result, csv_rows


def markdown(summary):
    lines = [
        "# KV AllGather ablation",
        "",
        "These are source-bound MHA operator measurements, not full-model results. All arms use the same inputs per case and rank. The attention backend is forced PyTorch Flash SDPA, with BF16, 32 Q/K/V heads, head dimension 128, and noncausal attention.",
        "",
        "Each timing sample is the mean per call within one round, followed by the maximum across ranks. Standard deviation and range describe the round means, not request latency percentiles. Positive latency reduction means faster; negative means slower.",
        "",
        "The four contrasts compare implementations. They are not independent additive attribution: packing, output strides, and memory copies can interact. Layout-only keeps two gathers; packing-only keeps batch-first gather layout; head-contiguous materializes the head's K/V views.",
        "",
    ]
    for run in summary["runs"]:
        meta = run["metadata"]
        lines += [
            f"## Run {run['run']}: {meta['world_size']} GPUs",
            "",
            f"Hardware: {meta['gpu']}. Python {meta['python']}; PyTorch {meta['torch']}; CUDA {meta['cuda']}; NCCL {'.'.join(map(str, meta['nccl']))}.",
            "",
            f"Base: `{meta['base_sha']}`. Head: `{meta['head_sha']}`.",
            "",
            f"Each arm: {meta['rounds']} rounds, {meta['warmup_per_sample']} warmups and {meta['iterations_per_sample']} measured calls per sample. Validation: **{run['validation']['status']}**. Source file hashes and individual samples are in `summary.json` and the raw artifacts.",
            "",
        ]
        for item in run["cases"]:
            case = item["case"]
            errors = list(item["validation"]["attention_max_abs_error"].values())
            graph_errors = list(item["changed_input_max_abs_error"].values())
            lines += [
                f"### B={case['batch']}, global S={case['seq']}, {case['layout']}",
                "",
                f"Exact Q/K/V equality: {item['validation']['qkv_exact']}. Largest attention error: {max(errors):g}; changed-input graph replay error: {max(graph_errors):g}.",
                "",
                "| Arm | Gather calls (expected) | Preparation peak increment, MiB | Retained increment, MiB |",
                "|---|---:|---:|---:|",
            ]
            for name in ARMS:
                arm = item["arms"][name]
                lines.append(
                    f"| {name} | {arm['all_gather_calls']} ({arm['expected_all_gather_calls']}) | {arm['pre_attention_peak_allocated_increment_mib']:.1f} | {arm['pre_attention_retained_increment_mib']:.1f} |"
                )
            lines += [
                "",
                "Memory is eager preparation's allocated increment over entry allocation, with validation outputs freed; it is not total model memory.",
                "",
            ]
            for stage in STAGES:
                lines += [
                    f"**{stage}**, GPU milliseconds",
                    "",
                    "| Arm | Mean | Sample std | Median | Min–max |",
                    "|---|---:|---:|---:|---:|",
                ]
                for name in ARMS:
                    val = item["arms"][name]["timings"][stage]["gpu_ms"]
                    lines.append(
                        f"| {name} | {val['mean']:.6f} | {val['sample_std']:.6f} | {val['median']:.6f} | {val['min']:.6f}–{val['max']:.6f} |"
                    )
                lines.append("")
            lines += [
                "| Contrast | Eager preparation | Eager preparation + attention | Graph preparation + attention |",
                "|---|---:|---:|---:|",
            ]
            for label, contrast in item["contrasts"].items():
                cells = [f"{contrast[stage]['latency_reduction_percent']:+.2f}%" for stage in STAGES]
                lines.append(f"| {label} | {' | '.join(cells)} |")
            lines.append("")
        profile = run["profile"]
        lines += [f"### Untimed profile: {profile['status']}", ""]
        if "arms" in profile:
            lines += [
                profile["note"],
                "",
                "| Arm | Status | NCCL CUDA kernel events | aten::clone | aten::copy_ | aten::contiguous |",
                "|---|---|---:|---:|---:|---:|",
            ]
            for name, arm in profile["arms"].items():
                counts = arm["aten_counts"]
                lines.append(
                    f"| {name} | {arm['status']} | {arm['nccl_kernel_count']} | {counts['aten::clone']} | {counts['aten::copy_']} | {counts['aten::contiguous']} |"
                )
            lines += [
                "",
                "NCCL kernel counts depend on the selected protocol and need not equal collective counts. Operator counts describe observed work; they do not establish how much latency each operation caused.",
                "",
            ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--outputdir", type=Path, required=True)
    args = parser.parse_args()
    summary = {"scope": "MHA operator ablation; not full-model inference", "runs": []}
    flat = []
    for run_dir in discover_runs(args.runs):
        run, csv_rows = summarize_run(run_dir)
        summary["runs"].append(run)
        flat.extend(csv_rows)
    if not flat:
        raise ValueError("No result rows available; no report was written")
    args.outputdir.mkdir(parents=True, exist_ok=True)
    (args.outputdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.outputdir / "summary.md").write_text(markdown(summary))
    with (args.outputdir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(
        json.dumps(
            {
                "runs": len(summary["runs"]),
                "case_arm_stage_rows": len(flat),
                "outputs": ["summary.json", "summary.md", "summary.csv"],
            }
        )
    )


if __name__ == "__main__":
    main()
