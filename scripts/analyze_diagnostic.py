#!/usr/bin/env python3
"""Audit unscored two-arm Klein profiler evidence; never emits a service score.

python analyze_diagnostic.py DIAGNOSTIC_RUN --scored-run 9723875
Requires analyze_service.py and Pillow. Reads only frozen artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path

import analyze_service as service

ARMS = ("control-a", "batch")
TARGETS = ["text_encoder.forward", "transformer.forward", "vae.encode", "vae.decode"]
REQUIRED = ("vae.decode", "transformer.forward")
HOOK = "vae_diagnostic_hooks/sitecustomize.py"
PREFIX = re.compile(r"\(DiffusionWorker_TP([0-3]) pid=(\d+)\)")
STAGE = re.compile(r"\[DiffusionPipelineProfiler\] Flux2KleinPipeline\.(\S+) took ([0-9.eE+-]+)s")
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def inspect_logs(path, hook_hash, rows):
    text = ANSI.sub("", path.read_text())
    ranks = {}
    for rank, pid in PREFIX.findall(text):
        service.require(pid not in ranks or ranks[pid] == int(rank), "Conflicting PID/rank mapping")
        ranks[pid] = int(rank)
    markers = []
    for match in re.finditer(r"OMNI_VAE_BATCH_DIAGNOSTIC_HOOK\s+", text):
        marker, _ = json.JSONDecoder().raw_decode(text[match.end() :])
        service.require(marker["hook_sha256"] == hook_hash and marker["targets"] == TARGETS, "Wrong hook proof/hash")
        service.require(
            marker["module"] == "vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein"
            and marker["module_file"].endswith("/vllm_omni/diffusion/models/flux2_klein/pipeline_flux2_klein.py"),
            "Hook attached to an unexpected module",
        )
        markers.append(marker)
    worker_markers = {str(marker["pid"]): marker for marker in markers if str(marker["pid"]) in ranks}
    service.require(
        len(worker_markers) == 4 and {ranks[pid] for pid in worker_markers} == {0, 1, 2, 3},
        "Four distinct worker hook receipts/rank mappings are required",
    )
    # Startup warmups must be excluded. Never infer rank from interleaved prefixes.
    lines = text.splitlines()
    ready = next((i for i, line in enumerate(lines) if 'GET /health HTTP/1.1" 200' in line), None)
    events = {rank: [] for rank in range(4)}
    unassigned = []
    for index, line in enumerate(lines):
        if ready is None or index <= ready:
            continue
        matches = list(STAGE.finditer(line))
        if not matches:
            continue
        prefixes = PREFIX.findall(line)
        for match in matches:
            target, seconds = match.groups()
            event = {"line": index + 1, "target": target, "wall_ms": float(seconds) * 1000}
            service.require(math.isfinite(event["wall_ms"]) and event["wall_ms"] >= 0, "Invalid profiler duration")
            if len(prefixes) == 1 and len(matches) == 1:
                events[int(prefixes[0][0])].append(event)
            else:
                unassigned.append({**event, "reason": "ambiguous or absent worker prefix"})
    per_rank = {}
    for rank, rank_events in events.items():
        groups, current = [], []
        for event in rank_events:
            current.append(event)
            if event["target"] == "forward":
                groups.append(current)
                current = []
        aligned = len(groups) == len(rows) and not current
        aligned = aligned and all(
            sum(e["target"] == "transformer.forward" for e in group) == 4
            and sum(e["target"] == "vae.decode" for e in group) == 1
            for group in groups
        )
        result = {"aligned": aligned, "forward_groups": len(groups), "raw_events": rank_events}
        if aligned:
            result["requests"] = [
                {
                    "label": row["label"],
                    "warmup": row["warmup"],
                    "transformer_call_count": 4,
                    "transformer_wall_ms": sum(e["wall_ms"] for e in group if e["target"] == "transformer.forward"),
                    "vae_decode_wall_ms": next(e["wall_ms"] for e in group if e["target"] == "vae.decode"),
                    "pipeline_forward_wall_ms": group[-1]["wall_ms"],
                }
                for row, group in zip(rows, groups)
            ]
        per_rank[str(rank)] = result
    return {
        "hook_markers": markers,
        "worker_pid_to_rank": ranks,
        "health_boundary_found": ready is not None,
        "per_rank": per_rank,
        "unassigned_events": unassigned,
        "all_ranks_aligned": all(value["aligned"] for value in per_rank.values()),
    }


def audit_diagnostic_arm(root, label, manifest, hashes):
    arm = service.audit_arm(root, label, manifest, hashes, diagnostic=True)
    requests, summaries = service.metric_observations(arm["requests"])
    for request in requests:
        for target in REQUIRED:
            key = f"stage_durations.Flux2KleinPipeline.{target}_ms"
            service.require(
                service.positive(request["corrected_server_time_metrics_ms"].get(key)),
                f"{label}/{request['label']}: positive raw {target} profiler stage required",
            )
            service.require(
                request["normalization_sources"][key]["unit_basis"] == "frozen_native_profiler",
                "Profiler unit is not source backed",
            )
    logs = inspect_logs(root / label / "server.log", hashes[HOOK], arm["requests"])
    # Check rank 0's four-call log sum against the returned profiler accumulation,
    # allowing the native logger's six-decimal seconds rounding.
    if logs["per_rank"]["0"]["aligned"]:
        for request, logged in zip(requests, logs["per_rank"]["0"]["requests"]):
            for target, field, tolerance in (
                ("transformer.forward", "transformer_wall_ms", 0.003),
                ("vae.decode", "vae_decode_wall_ms", 0.001),
            ):
                value = request["corrected_server_time_metrics_ms"][f"stage_durations.Flux2KleinPipeline.{target}_ms"]
                service.require(
                    abs(value - logged[field]) <= tolerance,
                    f"{label}/{request['label']}: HTTP/rank0 {target} log sum differs",
                )
    output = {
        "metadata": arm["metadata"],
        "repeat_checks": arm["repeat_checks"],
        "raw_original_corrected_metrics": requests,
        "stage_observations_ms": summaries,
        "log_evidence": logs,
    }
    if logs["all_ranks_aligned"]:
        output["largest_observed_rank_stage_ms"] = {}
        for batch in (2, 4):
            selected = [i for i, row in enumerate(arm["requests"]) if row["batch"] == batch and not row["warmup"]]
            output["largest_observed_rank_stage_ms"][str(batch)] = {}
            for field in ("transformer_wall_ms", "vae_decode_wall_ms"):
                samples = [
                    max(logs["per_rank"][str(rank)]["requests"][i][field] for rank in range(4)) for i in selected
                ]
                output["largest_observed_rank_stage_ms"][str(batch)][field] = {
                    "samples_ms": samples,
                    "mean_ms": statistics.mean(samples),
                    "sample_std_ms": statistics.stdev(samples),
                }
    return arm, output


def compare_scored(scored_root, diagnostic_arms, expected_manifest):
    harness = scored_root / "harness" if (scored_root / "harness").is_dir() else Path(__file__).parent
    manifest, hashes = service.audit_job(scored_root, expected_manifest, harness)
    scored = {label: service.audit_arm(scored_root, label, manifest, hashes) for label in service.ARMS}
    comparisons = {}
    for label, diagnostic in diagnostic_arms.items():
        for meta_key in ("source_sha", "source_sha256", "model_revision", "model_config_sha256", "versions"):
            service.require(
                diagnostic["metadata"][meta_key] == scored[label]["metadata"][meta_key],
                f"Scored/diagnostic {label} {meta_key} differs",
            )
        matched = []
        for row in diagnostic["requests"]:
            peer = next(other for other in scored[label]["requests"] if other["label"] == row["label"])
            service.require(row["payload"] == peer["payload"], "Scored/diagnostic payload differs")
            left, right = ([i["pixel_sha256"] for i in value["images"]] for value in (row, peer))
            matched.append(
                {
                    "label": row["label"],
                    "ordered_equal": left == right,
                    "diagnostic_hashes": left,
                    "scored_hashes": right,
                }
            )
        comparisons[label] = matched
    return {
        "run": str(scored_root),
        "all_ordered_hashes_exact": all(row["ordered_equal"] for rows in comparisons.values() for row in rows),
        "comparisons": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--scored-run", type=Path)
    parser.add_argument(
        "--expected-manifest", type=Path, default=Path(__file__).with_name("service_source_manifest.json")
    )
    parser.add_argument("--harness-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    harness = args.harness_dir or (root / "harness" if (root / "harness").is_dir() else Path(__file__).parent)
    report = {
        "status": "INCOMPLETE_OR_INVALID",
        "scored": False,
        "service_score_claim": False,
        "scope": "Native profiler synchronized host wall milliseconds, including VAE communication; not CUDA-event/kernel time or service latency score.",
        "analyzer_sha256": service.digest(Path(__file__)),
        "shared_analysis_sha256": service.digest(Path(service.__file__)),
        "errors": [],
        "warnings": [],
    }
    arms = {}
    try:
        manifest, hashes = service.audit_job(root, args.expected_manifest, harness, diagnostic=True)
        report.update(source_manifest=manifest, harness_sha256=hashes, arms={})
        for label in ARMS:
            arms[label], report["arms"][label] = audit_diagnostic_arm(root, label, manifest, hashes)
        for key in ("source_sha256", "model_config_sha256", "versions", "model_path", "pjm_job_id"):
            service.require(
                arms["control-a"]["metadata"][key] == arms["batch"]["metadata"][key],
                f"Across-arm metadata differs: {key}",
            )
        service.require(
            arms["control-a"]["metadata"]["ended_unix_s"] <= arms["batch"]["metadata"]["started_unix_s"],
            "Diagnostic arms overlap or are out of order",
        )
        for label in ARMS:
            if not report["arms"][label]["log_evidence"]["all_ranks_aligned"]:
                report["warnings"].append(
                    f"{label}: rank logs cannot prove four-call per-request alignment on every rank; no missing rank times inferred"
                )
        if args.scored_run:
            report["scored_image_comparison"] = compare_scored(args.scored_run.resolve(), arms, args.expected_manifest)
            service.require(
                report["scored_image_comparison"]["all_ordered_hashes_exact"], "Scored/diagnostic image hashes differ"
            )
        else:
            report["warnings"].append(
                "No scored-run comparison requested; scored/diagnostic image equivalence remains unverified"
            )
        report["status"] = "VALIDATED_DIAGNOSTIC_WITH_LIMITATIONS" if report["warnings"] else "VALIDATED_DIAGNOSTIC"
    except service.AUDIT_ERRORS as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    output = args.output or root / "diagnostic-analysis.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['status']}: {output}")
    for message in report["errors"] + report["warnings"]:
        print(message)
    return 1 if report["errors"] else 2 if report["warnings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
