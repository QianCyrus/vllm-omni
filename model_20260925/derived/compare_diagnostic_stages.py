#!/usr/bin/env python3
"""Compare one measured request from each frozen diagnostic run; never score it."""

import argparse
import hashlib
import json
import re
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def collect(run, derived):
    metadata = load(run / "metadata.json")
    validation = load(run / "validation.json")
    phases = load(derived / run.name / "diagnostic_phases.json")
    rows = [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines()]
    measured = [row for row in rows if not row["warmup"]]
    assert metadata["diagnostic"] and validation["status"] == "PASS"
    assert len(measured) == 1, "This comparison intentionally covers one diagnostic request"
    assert not phases["verification_problems"]
    assert len(measured[0]["stage_durations_raw"]) == 1
    rank0 = next(rank for rank in phases["ranks"] if rank["rank"] == 0)
    assert rank0["phases"]["measured_requests"]["forward_indices"] == [7, 8, 9, 10]
    structured = measured[0]["stage_durations_raw"][0]
    pipeline_values = phases["measured_pipeline_profiler_seconds_from_log"]["0"]["Flux2KleinPipeline.forward"]
    assert len(pipeline_values) == 1
    # Independent check: the only rank-0 pipeline line after driver warmup must
    # belong to this measured request. Reject multiplexed/ambiguous prefixes.
    log = (run / "job.log").read_text()
    measured_log = log.split("warmup_00 wall=", 1)[1].split("sample_00 wall=", 1)[0]
    pipeline_lines = [line for line in measured_log.splitlines()
                      if "Flux2KleinPipeline.forward took" in line and "DiffusionWorker_SP0 " in line]
    assert len(pipeline_lines) == 1
    assert len(re.findall(r"\(DiffusionWorker_SP\d+ pid=\d+\)", pipeline_lines[0])) == 1
    parsed = float(re.search(r"Flux2KleinPipeline.forward took ([\d.]+)s", pipeline_lines[0]).group(1))
    assert parsed == pipeline_values[0]
    stages = {
        "transformer_four_steps_cuda_ms": rank0["phases"]["measured_requests"]["transformer_cuda_ms_instrumented"]["sum"],
        "text_encoder_ms": structured["Flux2KleinPipeline.text_encoder.forward"] * 1000,
        "vae_decode_ms": structured["Flux2KleinPipeline.vae.decode"] * 1000,
        "pipeline_forward_ms": parsed * 1000,
        "request_wall_ms": measured[0]["wall_seconds"] * 1000,
    }
    for rank in phases["ranks"]:
        assert list(rank["pre_attention_calls_per_forward"].values()) == [25] * 10
        assert all(module["matches_metadata"] for module in rank["source_modules"])
        shapes = [item["representative"] for item in rank["representative_shapes"]
                  if item["representative"]["inputs"]["query"]["shape"][0] == 2]
        assert shapes and all(item["sp_size"] == 4 for item in shapes)
        for item in shapes:
            assert all(item["inputs"][name]["shape"] == [2, 1536, 24, 128] for name in ("query", "key", "value"))
            assert all(item["outputs"][name]["shape"] == [2, 6144, 24, 128] for name in ("key", "value"))
            assert all(value is None for value in item["joint"].values())
    terminate = re.findall(r"Calling terminate on diffusion worker ([^ ]+) \(pid=(\d+)\)", log)
    shutdown_ranks = sorted(set(int(rank) for rank in re.findall(r"Worker (\d+): Shutdown complete\.", log)))
    result = {
        "run_id": run.name, "source_sha": metadata["source_sha"], "stage_ms": stages,
        "phase_forward_indices": {name: value["forward_indices"] for name, value in rank0["phases"].items()},
        "measured_pre_attention_calls_per_rank": 100,
        "total_pre_attention_calls_per_rank": 250,
        "all_ranks_checked": [rank["rank"] for rank in phases["ranks"]],
        "artifact_manifest_files_verified": phases["artifact_manifest_files_verified"],
        "source_and_harness_verification": phases["source_and_harness_verification"],
        "cleanup": {
            "omni_close_returned_normally": validation["cleanup_complete"],
            "worker_python_cleanup_logged_ranks": shutdown_ranks,
            "terminate_warning_count": len(terminate),
            "kill_or_incomplete_cleanup_warning": bool(re.search(r"Calling kill on diffusion worker|Failed to terminate|cleanup incomplete", log, re.I)),
            "job_exit_code": int((run / "exit_code.txt").read_text()),
            "independent_worker_liveness_check": False,
        },
        "pixel_sha256": [image["pixel_sha256"] for image in measured[0]["images"]],
    }
    return result, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--outputdir", type=Path, required=True)
    args = parser.parse_args()
    out = args.outputdir.resolve()
    assert all(out != run.resolve() and run.resolve() not in out.parents for run in (args.base, args.head))
    base, base_meta = collect(args.base, out)
    head, head_meta = collect(args.head, out)
    keys = ("model_revision", "model_config_sha256", "world_size", "batch", "height", "width", "steps", "prompt", "seed", "attention_backend", "versions", "driver_sha256")
    mismatches = [key for key in keys if base_meta.get(key) != head_meta.get(key)]
    assert not mismatches, mismatches
    source_root = args.base.parent
    teardown_files = ["vllm_omni/diffusion/executor/multiproc_executor.py", "vllm_omni/diffusion/worker/diffusion_worker.py",
                      "vllm_omni/engine/omni_engine_base.py", "vllm_omni/engine/stage_pool.py"]
    teardown_sources = []
    for relative in teardown_files:
        a = (source_root / "source_base" / relative).read_bytes()
        b = (source_root / "source_head" / relative).read_bytes()
        assert a == b
        teardown_sources.append({"file": relative, "base_head_identical": True, "sha256": hashlib.sha256(a).hexdigest()})
    summary = {
        "diagnostic_only": True, "measured_requests_per_arm": 1, "rank_for_stage_table": 0,
        "base": base, "head": head, "matched_metadata_keys": list(keys),
        "native_attention_geometry": {"qkv_before_gather": [2, 1536, 24, 128], "kv_after_gather": [2, 6144, 24, 128], "joint_tensors": None},
        "timing_sources": {"transformer": "CUDA events in diagnostic hook, forwards 7–10 only",
                           "encoder_and_vae": "Measured row in results.jsonl, stage_durations_raw; seconds converted to milliseconds",
                           "pipeline": "Unambiguous rank-0 log line after warmup marker and before measured-request marker; checked against ordered profiler series",
                           "request": "Measured results.jsonl wall_seconds, converted to milliseconds"},
        "teardown_source_files": teardown_sources,
        "notes": ["One instrumented measured request per arm; this table is not a new scored speedup estimate.",
                  "Startup forwards 1–2 and request warmup forwards 3–6 are excluded from every stage value in this comparison.",
                  "Transformer, encoder and VAE timings are nested inside pipeline forward. Never add them to that enclosing total.",
                  "The native model gathers replicated text together with sharded image tokens; this does not validate the separate joint-text path or single-GPU equivalence.",
                  "Base shutdown logged no terminate or kill escalation. Head shutdown required terminate after the 15-second graceful wait; both Omni.close calls returned normally.",
                  "cleanup_complete means close returned without exception; the driver did not independently inspect worker liveness."]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Diagnostic stage comparison", "", "One instrumented measured request per arm, rank 0. This is stage attribution, not a new scored speedup estimate.", "",
             f"Base: `{base['source_sha']}` (run {base['run_id']}). Head: `{head['source_sha']}` (run {head['run_id']}).", "",
             "| Stage or enclosing total | Base, ms | Head, ms |", "| --- | ---: | ---: |"]
    labels = ("Transformer, four steps (CUDA events)", "Text encoder", "VAE decode", "Pipeline forward (enclosing total)", "Request wall (enclosing total)")
    for label, key in zip(labels, base["stage_ms"]):
        lines.append(f"| {label} | {base['stage_ms'][key]:.3f} | {head['stage_ms'][key]:.3f} |")
    lines.extend(["", "Encoder and VAE values come from structured measured results. Pipeline forward comes from validated rank-0 log lines. Transformer timings include diagnostic observer overhead.", "",
                  "Both runs: ranks 0–3; 25 target pre-attention calls per transformer forward; 50 startup, 100 warmup and 100 measured calls per rank. B2 Q/K/V are `[2,1536,24,128]`, gathered K/V are `[2,6144,24,128]`, and joint tensors are absent. Source hashes and artifact checks passed.", ""])
    lines.extend("- " + note for note in summary["notes"][1:])
    lines.extend(["", "Teardown implementations are byte-identical between the two source snapshots. The cleaner waits 15 seconds, then terminates surviving workers; it logs rather than raises if cleanup remains incomplete. Neither run logged kill or incomplete-cleanup warnings.", ""])
    (out / "stage_comparison.md").write_text("\n".join(lines))
    print(json.dumps({"base_stage_ms": base["stage_ms"], "head_stage_ms": head["stage_ms"], "outputdir": str(out)}))


if __name__ == "__main__":
    main()
