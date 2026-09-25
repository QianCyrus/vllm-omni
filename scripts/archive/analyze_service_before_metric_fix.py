#!/usr/bin/env python3
"""Audit the frozen control-a / batch / control-b HTTP service experiment.

Usage: python analyze_service.py RUN_DIRECTORY
Needs Pillow only for decoded RGB comparisons. No model imports or GPU work.
The scored protocol is B2/B4, two warmups and six samples, plus B1 and two
alternate prompt/seed requests. Invalid/incomplete evidence never gets a score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

ARMS = ("control-a", "batch", "control-b")
BASE = "43e507117f04a86f2df8d405cbe03c1b1642eac9"
HEAD = "1ccbc20901149b803a135984c2c3e4847fe981dd"
REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
PROMPT = "A small red sailboat on a calm blue lake, distant green hills, soft morning light, realistic photograph."
QUALITY = (
    "A ceramic blue teapot and two white cups on a wooden table, soft window light, realistic photograph.",
    "A yellow tram on a quiet city street after rain, detailed reflections, realistic photograph.",
)
SAVED_LABELS = ("b2-sample-00", "b4-sample-00", "b1-correctness", "b2-quality-00", "b2-quality-01")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def hash_ok(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def stats(samples):
    return {
        "count": len(samples), "samples_ms": samples,
        "mean_ms": statistics.mean(samples), "sample_std_ms": statistics.stdev(samples),
        "median_ms": statistics.median(samples), "min_ms": min(samples), "max_ms": max(samples),
    }


def expected_rows():
    rows = []
    for batch in (2, 4):
        for kind, count in (("warmup", 2), ("sample", 6)):
            rows.extend((f"b{batch}-{kind}-{i:02}", batch, kind == "warmup", False, PROMPT, 8137)
                        for i in range(count))
    rows.append(("b1-correctness", 1, False, True, PROMPT, 8137))
    rows.extend((f"b2-quality-{i:02}", 2, False, True, prompt, 8138 + i)
                for i, prompt in enumerate(QUALITY))
    return rows


def verify_ok_file(path, names):
    actual = path.read_text().splitlines()
    expected = [f"{name}: OK" for name in names]
    require(len(actual) == len(expected) and set(actual) == set(expected), f"{path}: verification entries differ")


def audit_job(root, expected_manifest, harness_dir):
    require((root / "exit_code.txt").read_text().strip() == "0", "Job exit code is not zero")
    require((root / "started_utc.txt").read_text().strip(), "Missing job start timestamp")
    require((root / "ended_utc.txt").read_text().strip(), "Missing job end timestamp")
    manifest = read_json(root / "source_manifest.json")
    require(manifest == read_json(expected_manifest), "Run source manifest differs from expected frozen manifest")
    require(manifest["head_sha"] == HEAD and manifest["base_sha"] == BASE, "Unexpected source head/base")
    require(hash_ok(manifest["patch_sha256"]), "Invalid patch hash")
    require(manifest["files"] and all(hash_ok(h) for h in manifest["files"].values()), "Invalid source file hashes")
    for name in ("source_verify_before.txt", "source_verify_after.txt"):
        verify_ok_file(root / name, manifest["files"])
    hashes = {}
    for line in (root / "harness_before.sha256").read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        require(match is not None, f"Malformed harness hash: {line}")
        value, name = match.groups()
        require(name not in hashes, f"Duplicate harness entry: {name}")
        hashes[name] = value
    names = {"launch_model_service.py", "bench_image_service.py", "run_service.pjm", "service_source_manifest.json"}
    require(set(hashes) == names, "Unexpected/missing harness hash entries")
    verify_ok_file(root / "harness_verify_after.txt", hashes)
    for name, value in hashes.items():
        path = root / "source_manifest.json" if name == "service_source_manifest.json" else harness_dir / name
        require(digest(path) == value, f"Archived harness hash mismatch: {path}")
    require(digest(root / "source_manifest.json") == digest(expected_manifest), "Source manifest byte hash differs")
    return manifest, hashes


def option(command, name):
    require(command.count(name) == 1, f"Missing/duplicate option {name}")
    return command[command.index(name) + 1]


def audit_arm(root, label, manifest, harness_hashes):
    from PIL import Image

    folder = root / label
    http = folder / "http"
    failure_files = list(folder.rglob("failure.json")) + list(http.glob("error-*")) + list(http.glob("invalid-*"))
    require(not failure_files, f"Failure artifacts exist: {failure_files}")
    meta = read_json(folder / "metadata.json")
    validation = read_json(folder / "validation.json")
    client = read_json(http / "metadata.json")
    client_validation = read_json(http / "validation.json")
    summary = read_json(http / "summary.json")
    for value in (meta, validation):
        require(value.get("status") == "PASS" and value.get("scored") is True
                and value.get("diagnostic") is False and value.get("source_unchanged") is True
                and value.get("cleanup_complete") is True, f"{label}: arm did not pass scored/source/cleanup validation")
    require(client_validation.get("status") == "PASS" and client_validation.get("scored") is True,
            f"{label}: HTTP validation failed or is unscored")
    arm = "batch" if label == "batch" else "control"
    require(meta["arm"] == client["arm"] == arm, f"{label}: wrong arm")
    require(meta["source_sha"] == client["source_sha"] == HEAD, f"{label}: wrong source SHA")
    require(meta["source_base_sha"] == BASE and meta["source_patch_sha256"] == manifest["patch_sha256"],
            f"{label}: source base/patch mismatch")
    require(meta["source_manifest_sha256"] == harness_hashes["service_source_manifest.json"], f"{label}: manifest hash mismatch")
    require(hash_ok(meta["source_sha256"].get("vllm_omni/diffusion/models/flux2_klein/flux2_klein_transformer.py")),
            f"{label}: actual Klein transformer source hash missing")
    require(all(meta["source_sha256"].get(name) == value for name, value in manifest["files"].items()),
            f"{label}: source file hashes mismatch")
    require(meta["launcher_sha256"] == harness_hashes["launch_model_service.py"], f"{label}: launcher mismatch")
    require(meta["client_sha256"] == client["client_sha256"] == harness_hashes["bench_image_service.py"],
            f"{label}: client mismatch")
    require(meta["model_revision"] == REVISION, f"{label}: wrong model revision")
    configs = {"model_index.json", "vae/config.json", "transformer/config.json", "text_encoder/config.json"}
    require(set(meta["model_config_sha256"]) == configs and all(hash_ok(h) for h in meta["model_config_sha256"].values()),
            f"{label}: model config hashes missing")
    require(meta["compile_mode"] == "regional" and meta["vae_mode"] == "batch"
            and meta["vae_degree"] == (4 if arm == "batch" else 1), f"{label}: wrong compile/VAE configuration")
    require(meta["parallelism"] == {"tp": 4, "sp": 1, "dp": 1, "pp": 1, "cfg": 1}, f"{label}: wrong parallelism")
    for key in ("client_cleanup", "server_cleanup"):
        require(meta[key]["started"] is True and meta[key]["group_alive"] is False, f"{label}: incomplete {key}")
    require(meta["client_cleanup"]["returncode"] == 0, f"{label}: client did not exit successfully")
    require(positive(meta["started_unix_s"]) and meta["ended_unix_s"] > meta["started_unix_s"], f"{label}: bad timestamps")
    command = meta["server_command"]
    expected_options = {
        "--model-class-name": "Flux2KleinPipeline", "--num-gpus": "4", "--dtype": "bfloat16",
        "--tensor-parallel-size": "4", "--data-parallel-size": "1", "--pipeline-parallel-size": "1",
        "--ulysses-degree": "1", "--ring-degree": "1", "--allgather-degree": "1", "--cfg-parallel-size": "1",
        "--vae-parallel-mode": "batch", "--vae-patch-parallel-size": str(meta["vae_degree"]),
        "--vae-fast-path": "off", "--cache-backend": "none", "--diffusion-attention-backend": "TORCH_SDPA",
        "--diffusion-compile-granularity": "regional",
    }
    require(all(option(command, k) == v for k, v in expected_options.items()), f"{label}: unexpected server command")
    require("--diffusion-compile-dynamic" in command and "--enable-diffusion-pipeline-profiler" not in command
            and "--enforce-eager" not in command, f"{label}: invalid scoring mode")
    expected_client = {"batches": [2, 4], "warmup_per_batch": 2, "repeats_per_batch": 6, "concurrency": 1,
                       "diagnostic": False, "b1_correctness_check": True, "alternate_prompt_seed_cases": 2}
    require(all(client.get(k) == v for k, v in expected_client.items()), f"{label}: client protocol mismatch")
    rows = [json.loads(line) for line in (http / "requests.jsonl").read_text().splitlines()]
    expected = expected_rows()
    require(len(rows) == client_validation["requests_completed"] == len(expected), f"{label}: missing/extra requests")
    require([row["label"] for row in rows] == [entry[0] for entry in expected], f"{label}: request order/labels differ")
    require(set(summary) == {"2", "4"}, f"{label}: summary batch keys differ")
    saved = {}
    for row, (name, batch, warmup, correctness, prompt, seed) in zip(rows, expected):
        where = f"{label}/{name}"
        require(row["arm"] == arm and row["batch"] == batch and row["warmup"] is warmup
                and row["correctness_only"] is correctness and row["scored"] is (not warmup and not correctness),
                f"{where}: incorrect request classification")
        payload = {"model": "klein-vae-bench", "prompt": prompt, "n": batch, "size": "1024x1024",
                   "num_inference_steps": 4, "guidance_scale": 1.0, "seed": seed, "generator_device": "cpu",
                   "response_format": "b64_json", "output_format": "png", "output_compression": 100,
                   "return_stage_metrics": True}
        require(row["payload"] == payload, f"{where}: payload mismatch")
        require(row["http_status"] == 200 and positive(row["wall_ms"]) and positive(row["response_bytes"]),
                f"{where}: failed HTTP or invalid latency/bytes")
        require(hash_ok(row["response_sha256"]), f"{where}: invalid response hash")
        require(isinstance(row["server_response_fields"], dict)
                and len(row["server_response_fields"].get("data", [])) == batch, f"{where}: response metadata/count missing")
        require(isinstance(row["server_time_metrics_ms"], dict), f"{where}: normalized metrics missing")
        require(len(row["images"]) == batch, f"{where}: incorrect image count")
        if name in SAVED_LABELS:
            saved[name] = []
        for index, item in enumerate(row["images"]):
            require(item["index"] == index and item["width"] == item["height"] == 1024 and item["mode"] == "RGB",
                    f"{where}: image index/shape/mode mismatch")
            require(hash_ok(item["pixel_sha256"]) and hash_ok(item["response_image_sha256"]), f"{where}: invalid image hashes")
            if name in SAVED_LABELS:
                relative = f"images/{name}-i{index}.png"
                require(item["file"] == relative, f"{where}: saved image path mismatch")
                path = http / relative
                require(digest(path) == item["saved_png_sha256"] == item["response_image_sha256"], f"{path}: PNG hash mismatch")
                with Image.open(path) as original:
                    require(original.format == "PNG", f"{path}: non-PNG image")
                    image = original.convert("RGB")
                    image.load()
                require(image.size == (1024, 1024), f"{path}: decoded image shape differs")
                require(hashlib.sha256(image.tobytes()).hexdigest() == item["pixel_sha256"], f"{path}: decoded RGB hash differs")
                saved[name].append(image)
            else:
                require(item["file"] is None and item["saved_png_sha256"] is None, f"{where}: unexpected saved artifact")
    require(len(list((http / "images").glob("*.png"))) == 11, f"{label}: missing/extra retained PNGs")
    timings = {}
    repeats = {}
    for batch in (2, 4):
        measured = [r for r in rows if r["batch"] == batch and r["scored"]]
        values = [r["wall_ms"] for r in measured]
        hashes = [[item["pixel_sha256"] for item in r["images"]] for r in measured]
        repeats[str(batch)] = all(value == hashes[0] for value in hashes)
        require(repeats[str(batch)], f"{label}: B{batch} same-seed repeat pixels differ")
        recorded = summary[str(batch)]
        require(recorded["scored"] is True and recorded["same_seed_repeat_pixels_exact"] is True
                and recorded["ordered_pixel_sha256"] == hashes[0] and recorded["samples_ms"] == values,
                f"{label}: B{batch} summary/repeat-check mismatch")
        timings[str(batch)] = stats(values)
        for key, computed in (("mean_ms", statistics.mean(values)), ("std_ms", statistics.stdev(values)),
                              ("median_ms", statistics.median(values)), ("min_ms", min(values)), ("max_ms", max(values))):
            require(math.isclose(recorded[key], computed, rel_tol=1e-12, abs_tol=1e-9), f"{label}: B{batch} {key} differs")
    return {"metadata": meta, "client_metadata": client, "requests": rows, "saved_images": saved,
            "latencies": timings, "repeat_checks": repeats}


def rgb_metrics(left, right):
    from PIL import ImageChops

    require(left.size == right.size and left.mode == right.mode == "RGB", "Cannot compare different RGB shapes")
    delta = ImageChops.difference(left, right)
    histogram = delta.histogram()
    channels = [sum(histogram[value::256]) for value in range(256)]
    channel_count = left.width * left.height * 3
    different_channels = channel_count - channels[0]
    mae = sum(value * count for value, count in enumerate(channels)) / channel_count
    mse = sum(value * value * count for value, count in enumerate(channels)) / channel_count
    rmse = math.sqrt(mse)
    red, green, blue = delta.split()
    any_channel = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    pixels = left.width * left.height
    return {
        "exact": different_channels == 0, "max_abs_0_255": max(i for i, count in enumerate(channels) if count),
        "mae_0_255": mae, "rmse_0_255": rmse,
        "psnr_db": "Infinity" if rmse == 0 else 20 * math.log10(255 / rmse),
        "different_channel_fraction": different_channels / channel_count,
        "different_pixel_fraction": (pixels - any_channel.histogram()[0]) / pixels,
    }


def compare(left, right):
    hashes = []
    for a, b in zip(left["requests"], right["requests"]):
        require(a["label"] == b["label"] and a["payload"] == b["payload"], "Cross-arm request/payload mismatch")
        ah = [i["pixel_sha256"] for i in a["images"]]
        bh = [i["pixel_sha256"] for i in b["images"]]
        hashes.append({"label": a["label"], "ordered_equal": ah == bh, "left": ah, "right": bh})
    pixels = {
        label: [{"index": i, **rgb_metrics(a, b)}
                for i, (a, b) in enumerate(zip(left["saved_images"][label], right["saved_images"][label]))]
        for label in SAVED_LABELS
    }
    return {"all_ordered_hashes_exact": all(row["ordered_equal"] for row in hashes),
            "ordered_hash_comparisons": hashes, "decoded_rgb_comparisons": pixels}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--expected-manifest", type=Path, default=Path(__file__).with_name("service_source_manifest.json"))
    parser.add_argument("--harness-dir", type=Path, help="Frozen matching launcher/client/PJM copies; defaults to RUN/harness or this script's directory")
    parser.add_argument("--output", type=Path, help="Defaults to RUN/service-analysis.json")
    args = parser.parse_args()
    root = args.run.resolve()
    output = args.output or root / "service-analysis.json"
    harness = args.harness_dir or (root / "harness" if (root / "harness").is_dir() else Path(__file__).parent)
    report = {"status": "INCOMPLETE_OR_INVALID", "scoring_eligible": False, "run": str(root),
              "analyzer_sha256": digest(Path(__file__)), "errors": [],
              "scope": "Sequential HTTP POST through complete response read; six requests per batch per process. No p99 estimate.",
              "std_definition": "Sample standard deviation (n-1), within one process; controls are never pooled.",
              "rgb_definition": "Decoded RGB uint8 channel values in [0,255]; differing pixel means any RGB channel differs; PSNR uses peak 255.",
              "metrics_policy": "Original server response fields and client-normalized values are preserved verbatim. Missing metrics stay absent; no inferred fields or units."}
    try:
        manifest, harness_hashes = audit_job(root, args.expected_manifest, harness)
        report["source_manifest"] = manifest
        report["harness_sha256"] = harness_hashes
    except Exception as exc:
        report["errors"].append(f"job: {type(exc).__name__}: {exc}")
        # Arm diagnostics may still be useful, but no scores follow a job failure.
        try:
            manifest = read_json(args.expected_manifest)
            harness_hashes = {name: digest(harness / name) for name in ("launch_model_service.py", "bench_image_service.py")
                              if (harness / name).is_file()}
            harness_hashes["service_source_manifest.json"] = digest(args.expected_manifest)
        except Exception as fallback_error:
            report["errors"].append(f"expected artifacts: {type(fallback_error).__name__}: {fallback_error}")
            manifest = None
    arms = {}
    for label in ARMS if manifest is not None else ():
        try:
            arms[label] = audit_arm(root, label, manifest, harness_hashes)
        except Exception as exc:
            report["errors"].append(f"{label}: {type(exc).__name__}: {exc}")
    if not report["errors"]:
        try:
            for key in ("source_sha256", "versions", "model_path", "model_config_sha256", "pjm_job_id"):
                require(all(arms[label]["metadata"][key] == arms["control-a"]["metadata"][key] for label in ARMS),
                        f"Across-arm metadata differs: {key}")
            for left, right in zip(ARMS, ARMS[1:]):
                require(arms[left]["metadata"]["ended_unix_s"] <= arms[right]["metadata"]["started_unix_s"],
                        f"Arms are overlapping/out of order: {left}/{right}")
            commands = []
            for label in ARMS:
                cmd = list(arms[label]["metadata"]["server_command"])
                cmd[cmd.index("--vae-patch-parallel-size") + 1] = "<degree>"
                commands.append(cmd)
            require(commands[0] == commands[1] == commands[2], "Server commands differ beyond VAE degree")
            pairs = (("control-a", "batch"), ("control-b", "batch"), ("control-a", "control-b"))
            comparisons = {f"{a}_vs_{b}": compare(arms[a], arms[b]) for a, b in pairs}
            reductions = {}
            for batch in ("2", "4"):
                candidate = arms["batch"]["latencies"][batch]["mean_ms"]
                baselines = {label: arms[label]["latencies"][batch]["mean_ms"] for label in ("control-a", "control-b")}
                reductions[batch] = {
                    "batch_minus_control_mean_ms": {label: candidate - value for label, value in baselines.items()},
                    "latency_reduction_pct_vs_control": {label: 100 * (1 - candidate / value) for label, value in baselines.items()},
                    "control_b_minus_a_pct_of_a": 100 * (baselines["control-b"] / baselines["control-a"] - 1),
                    "at_least_10pct_reduction_vs_each_control": all(candidate <= .9 * value for value in baselines.values()),
                }
            exact = all(value["all_ordered_hashes_exact"] for value in comparisons.values())
            report.update(status="COMPLETE_EXACT" if exact else "COMPLETE_NONEXACT_REVIEW_REQUIRED",
                          scoring_eligible=True, cross_arm_pixels_exact=exact,
                          latencies={label: arms[label]["latencies"] for label in ARMS},
                          repeat_checks={label: arms[label]["repeat_checks"] for label in ARMS},
                          reductions=reductions, comparisons=comparisons,
                          arm_metadata={label: arms[label]["metadata"] for label in ARMS},
                          timing_scope=arms["control-a"]["client_metadata"]["timing_scope"],
                          raw_server_metrics={label: [{"label": row["label"], "scored": row["scored"],
                              "server_response_fields": row["server_response_fields"],
                              "client_recorded_server_time_metrics_ms": row["server_time_metrics_ms"]}
                              for row in arms[label]["requests"]] for label in ARMS})
        except Exception as exc:
            report["errors"].append(f"cross-arm: {type(exc).__name__}: {exc}")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['status']}: {output}")
    for error in report["errors"]:
        print(error, file=sys.stderr)
    if report["scoring_eligible"]:
        for batch in ("2", "4"):
            text = "; ".join(f"{label} {arms[label]['latencies'][batch]['mean_ms']:.3f} ± "
                             f"{arms[label]['latencies'][batch]['sample_std_ms']:.3f} ms" for label in ARMS)
            change = report["reductions"][batch]["latency_reduction_pct_vs_control"]
            print(f"B{batch}: {text}; reduction vs control-a {change['control-a']:.2f}%, vs control-b {change['control-b']:.2f}%")
    return 0 if report["status"] == "COMPLETE_EXACT" else 2 if report["scoring_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
