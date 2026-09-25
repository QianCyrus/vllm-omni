#!/usr/bin/env python3
"""Derive phase-separated diagnostics from frozen full-model artifacts."""

import argparse
import collections
import hashlib
import json
import re
import statistics
from pathlib import Path


def stats(values):
    return {
        "samples": values,
        "sum": sum(values),
        "mean": statistics.mean(values) if values else None,
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "median": statistics.median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--outputdir", type=Path, required=True)
    parser.add_argument("--startup-forwards", type=int, default=2)
    args = parser.parse_args()
    run = args.run.resolve()
    output = args.outputdir.resolve()
    if output == run or run in output.parents:
        raise ValueError("Derived output must be outside the raw run directory")
    metadata = json.loads((run / "metadata.json").read_text())
    rows = read_jsonl(run / "results.jsonl")
    assert metadata["diagnostic"], "Expected a separate diagnostic run"
    warmup_count = sum(bool(row["warmup"]) for row in rows)
    measured_count = len(rows) - warmup_count
    assert warmup_count == metadata["warmup_requests"]
    assert measured_count == metadata["measured_requests"]
    steps = metadata["steps"]
    startup_end = args.startup_forwards
    warmup_end = startup_end + warmup_count * steps
    final_end = warmup_end + measured_count * steps
    phases = {
        "startup_dummy": (1, startup_end),
        "request_warmup": (startup_end + 1, warmup_end),
        "measured_requests": (warmup_end + 1, final_end),
    }
    problems = []
    manifest_count = 0
    for line in (run / "ARTIFACT_SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = run / name
        manifest_count += 1
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            problems.append("Artifact hash mismatch: " + name)
    source_verification = {}
    for filename in ("source_verify_before.txt", "source_verify_after.txt", "harness_verify_after.txt"):
        statuses = collections.Counter(line.rsplit(": ", 1)[-1] for line in (run / filename).read_text().splitlines())
        source_verification[filename] = dict(statuses)
        if set(statuses) != {"OK"}:
            problems.append("Verification failure: " + filename)
    ranks = []
    for path in sorted((run / "diagnostics").glob("events-*.jsonl")):
        events = read_jsonl(path)
        forwards = [e for e in events if e["event"] == "transformer_forward"]
        if not forwards:
            continue
        rank = forwards[0]["rank"]
        assert [e["call_index"] for e in forwards] == list(range(1, final_end + 1))
        attention = [e for e in events if e["event"] == "pre_attention"]
        assert all(e["status"] == "ok" and not e.get("observation_error") for e in attention)
        assert all(e["status"] == "ok" and e["timing_status"] == "cuda_events" for e in forwards)
        assert all(e.get("world_size") == metadata["world_size"] for e in attention + forwards)
        assert not any(e["event"] == "instrumentation_error" for e in events)
        hashes = []
        for event in events:
            if event["event"] != "module_patched":
                continue
            key = event["module"].replace(".", "/") + ".py"
            matches = metadata["source_sha256"].get(key) == event["source_sha256"]
            hashes.append({"module": event["module"], "sha256": event["source_sha256"], "matches_metadata": matches})
            if not matches:
                problems.append(f"Rank {rank}: source mismatch for {key}")
        phase_results = {}
        for label, (start, end) in phases.items():
            selected = [e for e in forwards if start <= e["call_index"] <= end]
            selected_attention = [e for e in attention if start <= e["forward"]["call_index"] <= end]
            phase_results[label] = {
                "forward_indices": [e["call_index"] for e in selected],
                "transformer_cuda_ms_instrumented": stats([e["cuda_ms"] for e in selected]),
                "pre_attention_calls": len(selected_attention),
            }
        shapes = [{"first_call": e["call_index"], "first_forward": e["forward"]["call_index"],
                   "representative": e["representative"]} for e in attention if e.get("representative")]
        ranks.append({"rank": rank, "phases": phase_results, "source_modules": hashes,
                      "pre_attention_calls_per_forward": dict(collections.Counter(e["forward"]["call_index"] for e in attention)),
                      "representative_shapes": shapes,
                      "all_forward_mean_ms_including_startup_and_warmup": statistics.mean(e["cuda_ms"] for e in forwards)})
    assert sorted(r["rank"] for r in ranks) == list(range(metadata["world_size"]))
    profiler = collections.defaultdict(lambda: collections.defaultdict(list))
    log_warnings = []
    pattern = re.compile(r"\(DiffusionWorker_SP(\d+) pid=\d+\).*?\[DiffusionPipelineProfiler\] (\S+) took ([\d.]+)s")
    for line_number, line in enumerate((run / "job.log").read_text().splitlines(), 1):
        match = pattern.search(line)
        if match:
            rank, method, seconds = match.groups()
            if len(re.findall(r"\(DiffusionWorker_SP\d+ pid=\d+\)", line)) != 1:
                log_warnings.append(f"Skipped ambiguous interleaved profiler prefixes at line {line_number}: {method}")
                continue
            profiler[int(rank)][method].append(float(seconds))
    measured_stage = {}
    expected_requests = 1 + len(rows)  # One engine dummy request precedes driver requests.
    for rank, methods in profiler.items():
        measured_stage[rank] = {}
        for method, values in methods.items():
            if len(values) == expected_requests:
                measured_stage[rank][method] = values[1 + warmup_count:]
            elif method == "Flux2KleinPipeline.forward":
                problems.append(f"Pipeline profiler entry count requires manual review: rank {rank}, {method}")
            elif method != "Flux2KleinPipeline.vae.encode":
                log_warnings.append(f"Omitted incomplete optional profiler series: rank {rank}, {method}")
    structured_stages = collections.defaultdict(list)
    for row in rows:
        if row["warmup"]:
            continue
        assert len(row["stage_durations_raw"]) == 1, "Expected a single pipeline stage"
        for name, value in row["stage_durations_raw"][0].items():
            if name.startswith("Flux2KleinPipeline."):
                structured_stages[name].append(value)
    summary = {
        "run_id": run.name, "source_sha": metadata["source_sha"], "diagnostic_only": True,
        "phase_rule": {"startup_forward_count": args.startup_forwards, "steps_per_driver_request": steps,
                       "evidence": "Engine _dummy_run uses two steps; driver runs warmups before measured requests. Validate ordered counts before slicing."},
        "artifact_manifest_files_verified": manifest_count, "verification_problems": problems,
        "source_and_harness_verification": source_verification, "ranks": sorted(ranks, key=lambda r: r["rank"]),
        "measured_pipeline_profiler_seconds_from_log": measured_stage,
        "structured_rank0_measured_stage_seconds": dict(structured_stages),
        "optional_profiler_log_warnings": log_warnings,
        "measured_request_wall_seconds": [r["wall_seconds"] for r in rows if not r["warmup"]],
        "notes": [
            "Transformer CUDA events include observer overhead and host-induced gaps. These are not uninstrumented performance scores.",
            "Transformer, text encoder and VAE timings are nested inside pipeline forward; never add them to that total.",
            "Representatives are deduplicated by tensor metadata across all calls; first-seen B2 shapes occur during request warmup.",
            "Observed B2 native path gathers sequence 1536 to 6144, with no joint tensors. It does not validate the separate joint-text path.",
            "A single measured diagnostic request is attribution evidence, not a speedup estimate or an independent quality comparison.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnostic_phases.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Phase-separated diagnostic", "", f"Run {run.name}; source `{metadata['source_sha']}`.", "",
             "Timings below are diagnostic CUDA-event measurements with instrumentation. They are not scored latency.", "",
             "| Rank | Startup dummy, sum ms | Request warmup, sum ms | Measured 4-step sum ms | Measured step mean ± sample std, ms |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for rank in summary["ranks"]:
        phase = rank["phases"]
        dummy, warm, measured = [phase[k]["transformer_cuda_ms_instrumented"] for k in phases]
        lines.append(f"| {rank['rank']} | {dummy['sum']:.3f} | {warm['sum']:.3f} | {measured['sum']:.3f} | {measured['mean']:.3f} ± {measured['sample_std']:.3f} |")
    lines.extend(["", "Forward indices: startup 1–2; request warmup 3–6; measured request 7–10. Each forward has 25 target pre-attention calls per rank.", "",
                  "The existing analyzer's all-forward average includes startup and warmup. Use this separated report for stage attribution.", "",
                  "Observed B2 Q/K/V shape before gathering: `[2,1536,24,128]`; gathered K/V: `[2,6144,24,128]`. Joint tensors are absent. Image tokens are sharded, while replicated text is concatenated before gathering in this native model path.", "",
                  "This confirms the target optimization runs. It supports a matched native base/head comparison; it does not establish single-GPU model equivalence or the separate joint-text path.", ""])
    lines.extend(["Measured request pipeline profiler (seconds; rank 0 encoder/VAE from structured results, other values from rounded logs):", "",
                  "| Rank | Pipeline forward | Text encoder | VAE decode |",
                  "| --- | ---: | ---: | ---: |"])
    for rank in sorted(measured_stage):
        values = dict(measured_stage[rank])
        if rank == 0:
            values.update(structured_stages)
        means = [statistics.mean(values["Flux2KleinPipeline." + method])
                 if "Flux2KleinPipeline." + method in values else None
                 for method in ("forward", "text_encoder.forward", "vae.decode")]
        cells = [f"{value:.6f}" if value is not None else "unavailable" for value in means]
        lines.append(f"| {rank} | {' | '.join(cells)} |")
    lines.extend(["", "Transformer, encoder and VAE measurements are nested inside pipeline forward. Do not add them to the pipeline total.", ""])
    lines.extend(summary["notes"])
    lines.extend(["", f"Artifact files checked: {manifest_count}. Verification problems: {len(problems)}.", ""])
    lines.extend(log_warnings)
    (output / "diagnostic_phases.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "problems": problems, "ranks": len(ranks)}))


if __name__ == "__main__":
    main()
