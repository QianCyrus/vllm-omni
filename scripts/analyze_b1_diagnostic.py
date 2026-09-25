#!/usr/bin/env python3
"""Audit the unscored 17-request B2/B4-history + unique B1 boundary probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import analyze_service as service

ARMS = ("control-a", "batch")
HOOK = "vae_b1_diagnostic_hooks/sitecustomize.py"
MODULES = (
    "vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_flux2",
    "vllm_omni.diffusion.models.flux2_klein.pipeline_flux2_klein",
)


def audit_receipts(root, harness):
    service.require((root / "exit_code.txt").read_text().strip() == "0", "Job did not complete successfully")
    manifest = service.read_json(root / "source_manifest.json")
    service.require(manifest["head_sha"] == service.HEAD and manifest["base_sha"] == service.BASE, "Unexpected source")
    service.require(
        manifest == service.read_json(Path(__file__).with_name("service_source_manifest.json")), "Manifest differs"
    )
    for name in ("source_verify_before.txt", "source_verify_after.txt"):
        service.verify_ok_file(root / name, manifest["files"])
    hashes = {}
    for line in (root / "harness_before.sha256").read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        service.require(match is not None, "Malformed harness receipt")
        value, name = match.groups()
        service.require(
            name not in hashes and not Path(name).is_absolute() and ".." not in Path(name).parts,
            "Duplicate or unsafe harness path",
        )
        path = root / "source_manifest.json" if name == "service_source_manifest.json" else harness / name
        service.require(service.digest(path) == value, f"Archived receipt differs: {name}")
        hashes[name] = value
    required = {
        "launch_model_service.py",
        "bench_image_service.py",
        "run_b1_diagnostic.pjm",
        "service_source_manifest.json",
        HOOK,
    }
    service.require(required <= set(hashes), "Required B1 harness receipts missing")
    service.verify_ok_file(root / "harness_verify_after.txt", hashes)
    return manifest, hashes


def audit_arm(root, label, manifest, hashes):
    from PIL import Image

    folder, expected_arm = root / label, "batch" if label == "batch" else "control"
    meta = service.read_json(folder / "metadata.json")
    val = service.read_json(folder / "validation.json")
    http = folder / "http"
    client = service.read_json(http / "metadata.json")
    http_val = service.read_json(http / "validation.json")
    summary = service.read_json(http / "summary.json")
    service.require(not list(folder.rglob("failure.json")), "Failure artifact exists")
    for value in (meta, val):
        service.require(
            value["status"] == "PASS"
            and value["scored"] is False
            and value["diagnostic"] is True
            and value["source_unchanged"]
            and value["cleanup_complete"],
            "Invalid diagnostic arm",
        )
    service.require(http_val["status"] == "PASS" and http_val["scored"] is False, "Invalid HTTP validation")
    service.require(
        meta["arm"] == client["arm"] == expected_arm and meta["source_sha"] == client["source_sha"] == service.HEAD,
        "Arm/source identity differs",
    )
    service.require(
        meta["source_manifest_sha256"] == hashes["service_source_manifest.json"]
        and meta["source_base_sha"] == service.BASE
        and meta["source_patch_sha256"] == manifest["patch_sha256"],
        "Source manifest/base/patch receipt differs",
    )
    service.require(
        all(meta["source_sha256"].get(path) == value for path, value in manifest["files"].items()),
        "Source file receipt differs",
    )
    service.require(
        meta["launcher_sha256"] == hashes["launch_model_service.py"]
        and meta["client_sha256"] == client["client_sha256"] == hashes["bench_image_service.py"],
        "Harness differs",
    )
    service.require(
        meta["compile_mode"] == "regional"
        and meta["vae_mode"] == "batch"
        and meta["vae_degree"] == (4 if label == "batch" else 1)
        and meta["parallelism"] == {"tp": 4, "sp": 1, "dp": 1, "pp": 1, "cfg": 1},
        "Configuration differs",
    )
    expected_client = {
        "batches": [2, 4],
        "warmup_per_batch": 2,
        "repeats_per_batch": 6,
        "concurrency": 1,
        "diagnostic": True,
        "b1_correctness_check": True,
        "alternate_prompt_seed_cases": 0,
    }
    service.require(
        all(client.get(k) == v for k, v in expected_client.items()), "Expected original 16-request B2/B4 history"
    )
    rows = [json.loads(line) for line in (http / "requests.jsonl").read_text().splitlines()]
    expected = service.expected_rows()[:17]
    service.require(len(rows) == http_val["requests_completed"] == 17, "Expected 17 complete requests")
    service.require([r["label"] for r in rows] == [r[0] for r in expected], "Request history/order differs")
    for row, (name, batch, warmup, correctness, prompt, seed) in zip(rows, expected):
        service.require(
            row["http_status"] == 200
            and row["scored"] is False
            and row["batch"] == batch
            and row["warmup"] is warmup
            and row["correctness_only"] is correctness,
            "Request classification differs",
        )
        expected_payload = {
            "model": "klein-vae-bench",
            "prompt": prompt,
            "n": batch,
            "size": "1024x1024",
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "seed": seed,
            "generator_device": "cpu",
            "response_format": "b64_json",
            "output_format": "png",
            "output_compression": 100,
            "return_stage_metrics": True,
        }
        service.require(row["payload"] == expected_payload and len(row["images"]) == batch, "Payload/count differs")
        for index, item in enumerate(row["images"]):
            service.require(
                item["index"] == index
                and item["width"] == item["height"] == 1024
                and item["mode"] == "RGB"
                and service.hash_ok(item["pixel_sha256"]),
                "Image metadata differs",
            )
            if name in ("b2-sample-00", "b4-sample-00", "b1-correctness"):
                relative = f"images/{name}-i{index}.png"
                service.require(item["file"] == relative, "Unexpected image path")
                path = http / relative
                service.require(
                    service.digest(path) == item["saved_png_sha256"] == item["response_image_sha256"],
                    "PNG hash differs",
                )
                with Image.open(path) as original:
                    image = original.convert("RGB")
                    image.load()
                service.require(
                    image.size == (1024, 1024) and hashlib.sha256(image.tobytes()).hexdigest() == item["pixel_sha256"],
                    "Decoded RGB hash differs",
                )
    service.require(len(list((http / "images").glob("*.png"))) == 7, "Retained image count differs")
    for batch in (2, 4):
        selected = [r for r in rows if r["batch"] == batch and not r["warmup"]]
        ordered = [[i["pixel_sha256"] for i in r["images"]] for r in selected]
        service.require(
            len(ordered) == 6
            and all(h == ordered[0] for h in ordered)
            and summary[str(batch)]["same_seed_repeat_pixels_exact"] is True
            and summary[str(batch)]["scored"] is False,
            "B2/B4 history repeats differ",
        )
    log = (folder / "server.log").read_text()
    ids = re.findall(r"RequestE2EStats \[request_id=(img_gen-[^\]]+)\]", log)
    service.require(len(ids) == len(set(ids)) == 17, "Cannot map 17 sequential HTTP requests to unique server IDs")
    target_id = ids[16]
    markers = []
    for match in re.finditer(r"OMNI_VAE_B1_DIAGNOSTIC_HOOK\s+", log):
        marker, _ = json.JSONDecoder().raw_decode(log[match.end() :])
        markers.append(marker)
        service.require(marker["hook_sha256"] == hashes[HOOK] and marker["module"] in MODULES, "Wrong B1 hook marker")
        source_path = marker["module"].replace(".", "/") + ".py"
        service.require(
            marker["module_sha256"] == meta["source_sha256"][source_path], "Hook module source hash differs"
        )
    native_hashes = {m.get("native_class_sha256") for m in markers if m["module"] == MODULES[0]}
    service.require(
        len(native_hashes) == 1 and all(service.hash_ok(value) for value in native_hashes),
        "Native class source receipt missing or inconsistent",
    )
    worker_pids = set(re.findall(r"\(DiffusionWorker_TP[0-3] pid=(\d+)\)", log))
    for module in MODULES:
        service.require(
            len({str(m["pid"]) for m in markers if m["module"] == module and str(m["pid"]) in worker_pids}) == 4,
            f"Four worker receipts missing: {module}",
        )
    captures_path = folder / "vae_boundary/rank0-b1-captures.jsonl"
    captures = [json.loads(line) for line in captures_path.read_text().splitlines()]
    selected = [r for r in captures if (r.get("request") or {}).get("request_id") == target_id]
    service.require(len(selected) == 1, "Unique real B1 request capture missing or duplicated")
    capture = selected[0]
    request = capture["request"]
    service.require(
        request["seed"] == 8137
        and request["num_inference_steps"] == 4
        and request["num_outputs_per_prompt"] == 1
        and request["generator_device"] == "cpu",
        "Captured B1 request parameters differ",
    )
    service.require(
        capture["status"] == "CAPTURED"
        and capture["rank"] == 0
        and capture["scored"] is False
        and capture["hook_sha256"] == hashes[HOOK]
        and capture["vae_parallel_degree"] == meta["vae_degree"],
        "Invalid B1 capture receipt",
    )
    service.require(
        capture["input"] == capture["input_after_fallback"] == capture["input_after_direct"], "Input was mutated"
    )
    service.require(capture["fallback_vs_direct_native"]["finite"] is True, "Native replay difference is non-finite")
    replay = capture["fallback_vs_direct_native"]
    service.require(
        replay["exact"] is (capture["fallback_output"] == capture["direct_native_output"]),
        "Native replay exactness disagrees with raw tensor hashes",
    )
    if replay["exact"]:
        service.require(all(replay[key] == 0 for key in ("max_abs", "mae", "rmse")), "Exact replay has nonzero error")
    if "input_tensor_file" in capture:
        tensor = captures_path.parent / capture["input_tensor_file"]
        service.require(
            tensor.resolve().is_relative_to(captures_path.parent.resolve())
            and service.digest(tensor) == capture["input_tensor_file_sha256"],
            "Saved latent receipt differs",
        )
    for record in captures:
        path = captures_path.parent / f"{record['capture_id']}.json"
        service.require(service.read_json(path) == record, "Capture manifest/individual record differs")
    return {
        "metadata": meta,
        "b1_request_id": target_id,
        "http_b1_pixel_sha256": rows[16]["images"][0]["pixel_sha256"],
        "request_sequence": [{"label": row["label"], "request_id": rid} for row, rid in zip(rows, ids)],
        "b1_capture": capture,
        "native_class_source_sha256": next(iter(native_hashes)),
        "hook_markers": markers,
        "excluded_captures": [
            {
                "reason": "Request ID is not the unique HTTP B1 correctness request; excluded from causal comparison",
                "capture": record,
            }
            for record in captures
            if record is not capture
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--harness-dir", type=Path)
    parser.add_argument("--scored-run", type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    harness = args.harness_dir or root / "harness"
    result = {
        "status": "INCOMPLETE_OR_INVALID",
        "scored": False,
        "errors": [],
        "analyzer_sha256": service.digest(Path(__file__)),
        "shared_analysis_sha256": service.digest(Path(service.__file__)),
        "scope": "VAE boundary and same-input native replay after original B2/B4 request history; no performance score or universal quality claim",
    }
    try:
        manifest, hashes = audit_receipts(root, harness)
        arms = {label: audit_arm(root, label, manifest, hashes) for label in ARMS}
        result.update(arms=arms, source_manifest=manifest, harness_sha256=hashes)
        for key in ("source_sha256", "model_revision", "model_config_sha256", "versions", "parallelism", "pjm_job_id"):
            service.require(
                arms["control-a"]["metadata"][key] == arms["batch"]["metadata"][key], f"Cross-arm {key} differs"
            )
        service.require(
            arms["control-a"]["native_class_source_sha256"] == arms["batch"]["native_class_source_sha256"],
            "Across-arm native Diffusers source differs",
        )
        left, right = (arms[label]["b1_capture"] for label in ARMS)
        same_input = left["input"] == right["input"]
        same_output = left["fallback_output"] == right["fallback_output"]
        same_rgb = arms["control-a"]["http_b1_pixel_sha256"] == arms["batch"]["http_b1_pixel_sha256"]
        result["boundary_comparison"] = {
            "input_exact": same_input,
            "raw_fallback_output_exact": same_output,
            "rgb_exact": same_rgb,
        }
        result["localization"] = (
            "DIVERGENCE_PRESENT_BEFORE_VAE"
            if not same_input
            else "DIVERGENCE_AT_VAE_EXECUTION"
            if not same_output
            else "DIVERGENCE_AFTER_VAE"
            if not same_rgb
            else "NO_CROSS_ARM_DIVERGENCE_OBSERVED"
        )
        result["native_replay_exact_each_arm"] = {
            label: arms[label]["b1_capture"]["fallback_vs_direct_native"]["exact"] for label in ARMS
        }
        if args.scored_run:
            historical = {}
            for label in service.ARMS:
                rows = [
                    json.loads(line)
                    for line in (args.scored_run / label / "http/requests.jsonl").read_text().splitlines()
                ]
                row = next(r for r in rows if r["label"] == "b1-correctness")
                historical[label] = row["images"][0]["pixel_sha256"]
            result["historical_scored_b1_hash_matches"] = {
                label: {peer: arm["http_b1_pixel_sha256"] == value for peer, value in historical.items()}
                for label, arm in arms.items()
            }
        result["status"] = "COMPLETE_BOUNDARY_EVIDENCE"
    except service.AUDIT_ERRORS as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    output = root / "b1-diagnostic-analysis.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{result['status']}: {output}")
    if "localization" in result:
        print(result["localization"])
    for error in result["errors"]:
        print(error)
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
