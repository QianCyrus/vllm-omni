#!/usr/bin/env python3
"""Read frozen full-model artifacts; keep every A/B/A run separate."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

BASE_SHA = "3d43571b94f1683412023c45bd886f9ce30f76bd"
HEAD_SHA = "a2cae127c2933f0e754d38d3615da58660c952f1"
WORKLOAD_KEYS = (
    "model",
    "model_revision",
    "model_config_sha256",
    "world_size",
    "batch",
    "height",
    "width",
    "steps",
    "prompt",
    "seed",
    "generator_policy",
    "attention_backend",
    "transformer_config",
)


def read_json(path):
    return json.loads(path.read_text())


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stats(values):
    if not values:
        return None
    return {
        "samples": values,
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def discover(paths):
    result = []
    for path in paths:
        candidates = (
            [path] if (path / "metadata.json").exists() else sorted(p.parent for p in path.rglob("metadata.json"))
        )
        for directory in candidates:
            if (directory / "results.jsonl").exists() and directory.resolve() not in result:
                result.append(directory.resolve())
    if not result:
        raise ValueError("No model runs found; no results were written")
    return result


def diagnostic_summary(directory, metadata):
    grouped, patches, pid_ranks = defaultdict(list), defaultdict(list), {}
    for path in sorted(directory.rglob("events-*.jsonl")):
        for event in read_lines(path):
            pid = event.get("pid")
            if event.get("rank") is not None:
                pid_ranks[pid] = event["rank"]
            if event.get("event") == "module_patched":
                module = event.get("module", "")
                expected = metadata.get("source_sha256", {}).get(module.replace(".", "/") + ".py")
                patches[pid].append(
                    {
                        "module": module,
                        "sha256": event.get("source_sha256"),
                        "matches_run_source": expected is not None and expected == event.get("source_sha256"),
                    }
                )
            grouped[pid].append(event)
    ranks = []
    for pid, events in grouped.items():
        route = [event for event in events if event.get("event") == "pre_attention"]
        forwards = [event for event in events if event.get("event") == "transformer_forward"]
        counts = Counter(
            str(event.get("forward", {}).get("call_index")) if event.get("forward") else "unlinked" for event in route
        )
        ranks.append(
            {
                "rank": pid_ranks.get(pid),
                "pid": pid,
                "source_modules": patches[pid],
                "pre_attention_calls": len(route),
                "pre_attention_status": dict(Counter(e.get("status") for e in route)),
                "pre_attention_calls_per_forward": dict(counts),
                "representative_shapes": [e["representative"] for e in route if e.get("representative")],
                "transformer_forward_calls": len(forwards),
                "transformer_cuda_ms_instrumented": stats([e["cuda_ms"] for e in forwards if "cuda_ms" in e]),
                "timing_status": dict(Counter(e.get("timing_status") for e in forwards)),
                "instrumentation_errors": [
                    e.get("operation", e.get("error_type")) for e in events if e.get("event") == "instrumentation_error"
                ],
            }
        )
    return {
        "status": "RECORDED" if grouped else "NO_HOOK_ARTIFACT",
        "processes": ranks,
        "note": "Counts include startup/warmup. Transformer timings include instrumentation, are not scored, and are nested inside pipeline forward; do not add them to pipeline totals.",
    }


def load_run(directory, base_sha, head_sha):
    meta = read_json(directory / "metadata.json")
    validation = (
        read_json(directory / "validation.json") if (directory / "validation.json").exists() else {"status": "MISSING"}
    )
    rows = read_lines(directory / "results.jsonl")
    measured = [row for row in rows if not row.get("warmup", False)]
    diagnostic = bool(meta.get("diagnostic"))
    source = meta.get("source_sha")
    role = "base" if source == base_sha else "head" if source == head_sha else "other"
    config = {key: meta.get(key) for key in WORKLOAD_KEYS}
    config.update({key: meta.get(key) for key in ("driver_sha256", "python", "warmup_requests", "measured_requests")})
    config["dependency_versions"] = {
        key: value for key, value in meta.get("versions", {}).items() if key != "vllm-omni"
    }
    config["engine_args"] = {
        key: value for key, value in meta.get("engine_args", {}).items() if key != "enable_diffusion_pipeline_profiler"
    }
    pixel_sets = [[image.get("pixel_sha256") for image in row.get("images", [])] for row in measured]
    run = {
        "run": directory.parent.name + "/" + directory.name,
        "role": role,
        "diagnostic": diagnostic,
        "metadata": meta,
        "validation": validation,
        "workload_config": config,
        "scored": not diagnostic and not meta.get("diagnostic_hooks_enabled", False),
        "complete": validation.get("status") == "PASS" and len(measured) == meta.get("measured_requests"),
        "wall_seconds": stats([row["wall_seconds"] for row in measured]),
        "warmup_wall_seconds": [row["wall_seconds"] for row in rows if row.get("warmup", False)],
        "repeated_pixel_hashes_equal": bool(pixel_sets) and all(values == pixel_sets[0] for values in pixel_sets),
        "stage_durations_raw": [row.get("stage_durations_raw", []) for row in measured],
        "worker_peak_reserved_mib": [row.get("worker_peak_reserved_mib", []) for row in measured],
        "diagnostic_hooks": diagnostic_summary(directory, meta),
    }
    return run, measured


def compare_pixels(left_dir, right_dir, left_rows, right_rows, metrics):
    issues, differences, checks, pixel_metrics = [], [], 0, []
    if not left_rows or len(left_rows) != len(right_rows):
        issues.append("Missing or unequal measured request counts")
    for sample, (left, right) in enumerate(zip(left_rows, right_rows)):
        old, new = left.get("images", []), right.get("images", [])
        if not old or len(old) != len(new):
            issues.append(f"sample {sample}: missing or unequal images")
        for index, (a, b) in enumerate(zip(old, new)):
            checks += 1
            paths = [left_dir / a.get("file", ""), right_dir / b.get("file", "")]
            valid = True
            for path, image in zip(paths, (a, b)):
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != image.get("png_sha256"):
                    issues.append(f"sample {sample}, image {index}: missing PNG or file hash mismatch")
                    valid = False
            if not a.get("pixel_sha256") or a.get("pixel_sha256") != b.get("pixel_sha256"):
                differences.append({"sample": sample, "image": index})
            if metrics is not None and valid:
                Image, np = metrics
                with Image.open(paths[0]) as image_a, Image.open(paths[1]) as image_b:
                    pixels_a, pixels_b = np.asarray(image_a.convert("RGB")), np.asarray(image_b.convert("RGB"))
                if pixels_a.shape != pixels_b.shape:
                    issues.append(f"sample {sample}, image {index}: unequal pixel shapes")
                    continue
                delta = pixels_a.astype(np.float64) - pixels_b.astype(np.float64)
                pixel_metrics.append(
                    {
                        "sample": sample,
                        "image": index,
                        "max_abs_uint8": float(np.abs(delta).max()),
                        "mean_abs_uint8": float(np.abs(delta).mean()),
                        "rmse_uint8": float(np.sqrt((delta * delta).mean())),
                        "different_channel_fraction": float((delta != 0).mean()),
                    }
                )
    exact = bool(checks) and not issues and not differences
    return {
        "status": "PIXEL_HASH_MATCH" if exact else "STRICT_EQUIVALENCE_UNVERIFIED",
        "image_pairs": checks,
        "png_files_verified": not issues and bool(checks),
        "recorded_pixel_hashes_exact": exact,
        "issues": issues,
        "pixel_hash_mismatches": differences,
        "pixel_metrics": pixel_metrics,
        "quality_note": "Pixel equality concerns these exact outputs only; general image quality was not evaluated.",
    }


def compare_runs(base, head, base_dir, head_dir, base_rows, head_rows, metrics):
    mismatches = [
        key for key in base["workload_config"] if base["workload_config"][key] != head["workload_config"].get(key)
    ]
    base_sources, head_sources = base["metadata"].get("source_sha256", {}), head["metadata"].get("source_sha256", {})
    changed_files = [
        key
        for key in sorted(base_sources.keys() | head_sources.keys())
        if base_sources.get(key) != head_sources.get(key)
    ]
    unexpected_sources = [
        key for key in changed_files if key != "vllm_omni/diffusion/attention/parallel/allgather_kv.py"
    ]
    valid = (
        not mismatches
        and not unexpected_sources
        and base["complete"]
        and head["complete"]
        and base["scored"]
        and head["scored"]
    )
    result = {
        "base_run": base["run"],
        "head_run": head["run"],
        "comparable_scored_runs": valid,
        "config_mismatch_keys": mismatches,
        "source_diff_files": changed_files,
        "unexpected_source_diff_files": unexpected_sources,
        "pixels": compare_pixels(base_dir, head_dir, base_rows, head_rows, metrics),
    }
    if valid and base["wall_seconds"] and head["wall_seconds"]:
        before, after = base["wall_seconds"]["mean"], head["wall_seconds"]["mean"]
        result["latency_reduction_percent"] = (before - after) / before * 100
        result["saved_seconds"] = before - after
    return result


def render(summary):
    lines = [
        "# Full-model A/B results",
        "",
        "Each run is reported separately. A/B/A baselines are not pooled. Positive reduction means lower head latency; negative means slower. Spread is across measured requests within a run, not independent job repetitions.",
        "",
        "| Run | Role | Kind | Validation | Mean ± std, s | Median | Min–max |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for run in summary["runs"]:
        value = run["wall_seconds"]
        cells = (
            f"{value['mean']:.6f} ± {value['sample_std']:.6f} | {value['median']:.6f} | {value['min']:.6f}–{value['max']:.6f}"
            if value
            else "— | — | —"
        )
        lines.append(
            f"| {run['run']} | {run['role']} | {'scored' if run['scored'] else 'diagnostic'} | {run['validation'].get('status')} | {cells} |"
        )
    lines += [
        "",
        "## Head versus each baseline",
        "",
        "| Baseline | Head | Config/completion check | Mean latency reduction | Output check |",
        "|---|---|---|---:|---|",
    ]
    for pair in summary["comparisons"]:
        delta = (
            f"{pair['latency_reduction_percent']:+.2f}%" if "latency_reduction_percent" in pair else "not comparable"
        )
        lines.append(
            f"| {pair['base_run']} | {pair['head_run']} | {pair['comparable_scored_runs']} | {delta} | {pair['pixels']['status']} |"
        )
    lines += [
        "",
        "A timing result does not establish output equivalence. Missing images, PNG hash failures, and pixel hash differences leave strict equivalence unverified. General image quality is not assessed. Config differences and optional pixel metrics are in `summary.json`.",
        "",
        "## Diagnostic route and stages",
        "",
        "Transformer timings are instrumented and nested inside pipeline forward. Do not add them to pipeline totals or use them as scored latency. Hook counts include startup and warmup. `stage_durations_raw` retains original names and mixed units; no stage values are blindly summed.",
        "",
    ]
    for run in summary["runs"]:
        if not run["diagnostic"]:
            continue
        lines += [
            f"**{run['run']}**: {run['diagnostic_hooks']['status']}",
            "",
            "| Rank | AllGather-KV preparation calls | Transformer calls | CUDA time mean per observed forward, ms | Source hashes match |",
            "|---|---:|---:|---:|---|",
        ]
        for process in run["diagnostic_hooks"]["processes"]:
            value = process["transformer_cuda_ms_instrumented"]
            hashes = process["source_modules"]
            lines.append(
                f"| {process['rank']} | {process['pre_attention_calls']} | {process['transformer_forward_calls']} | {value['mean'] if value else '—'} | {all(item['matches_run_source'] for item in hashes) if hashes else 'unverified'} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--outputdir", type=Path, required=True)
    parser.add_argument("--base-sha", default=BASE_SHA)
    parser.add_argument("--head-sha", default=HEAD_SHA)
    parser.add_argument("--pixel-metrics", action="store_true")
    args = parser.parse_args()
    metrics, metrics_status = None, "NOT_REQUESTED"
    if args.pixel_metrics:
        try:
            import numpy as np
            from PIL import Image

            metrics, metrics_status = (Image, np), "AVAILABLE"
        except ImportError:
            metrics_status = "PIL_OR_NUMPY_UNAVAILABLE"
    directories = discover(args.runs)
    loaded = [load_run(directory, args.base_sha, args.head_sha) for directory in directories]
    summary = {
        "runs": [run for run, _ in loaded],
        "comparisons": [],
        "pixel_metrics_status": metrics_status,
        "baseline_policy": "Every baseline is kept separate, including A/B/A before/after runs",
    }
    for bi, (base, base_rows) in enumerate(loaded):
        for hi, (head, head_rows) in enumerate(loaded):
            if base["role"] == "base" and head["role"] == "head" and base["scored"] and head["scored"]:
                summary["comparisons"].append(
                    compare_runs(base, head, directories[bi], directories[hi], base_rows, head_rows, metrics)
                )
    args.outputdir.mkdir(parents=True, exist_ok=True)
    (args.outputdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.outputdir / "summary.md").write_text(render(summary))
    print(
        json.dumps({"runs": len(loaded), "comparisons": len(summary["comparisons"]), "pixel_metrics": metrics_status})
    )


if __name__ == "__main__":
    main()
