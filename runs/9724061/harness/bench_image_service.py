#!/usr/bin/env python3
"""Explicit-n HTTP image benchmark; no model imports and no GPU work in this client."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import statistics
import time
import traceback
import urllib.error
import urllib.request

PROMPT = "A small red sailboat on a calm blue lake, distant green hills, soft morning light, realistic photograph."
QUALITY_PROMPTS = (
    "A ceramic blue teapot and two white cups on a wooden table, soft window light, realistic photograph.",
    "A yellow tram on a quiet city street after rain, detailed reflections, realistic photograph.",
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def time_metrics_ms(metrics):
    """Normalize explicit units and source-backed Klein profiler seconds only.

    stage_durations contains mixed units: explicit *_ms fields are already ms.
    Unknown unsuffixed stages remain in raw response metadata, unnormalized.
    """
    normalized = {}
    profiler_seconds = {
        f"Flux2KleinPipeline.{target}"
        for target in ("forward", "text_encoder.forward", "transformer.forward", "vae.encode", "vae.decode")
    }

    def visit(value, prefix="", parent=""):
        if not isinstance(value, dict):
            return
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(child, dict):
                visit(child, path, key)
            elif isinstance(child, (int, float)) and not isinstance(child, bool):
                if key.endswith("_ms"):
                    destination, factor = path, 1.0
                elif key.endswith("_s"):
                    destination, factor = path[:-2] + "_ms", 1000.0
                elif parent == "stage_durations" and key in profiler_seconds:
                    destination, factor = path + "_ms", 1000.0
                else:
                    continue
                if destination in normalized:
                    raise ValueError(f"Ambiguous normalized metric path: {destination}")
                normalized[destination] = float(child) * factor

    visit(metrics)
    return normalized


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8091")
    parser.add_argument("--model", default="klein-vae-bench")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[2])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--seed", type=int, default=8137)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--skip-b1-check", action="store_true")
    parser.add_argument("--quality-cases", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--diagnostic", action="store_true")
    return parser.parse_args()


def run(args):
    if any(batch < 1 or batch > 10 for batch in args.batches):
        raise ValueError("Batch sizes must be in the Images API range 1..10")
    if len(set(args.batches)) != len(args.batches):
        raise ValueError("Do not repeat a batch size in one client run")
    if args.warmup < 0 or args.repeats < 1 or args.steps < 1 or args.request_timeout <= 0:
        raise ValueError("Invalid warmup, repeats, steps or timeout")
    if args.height <= 0 or args.width <= 0 or args.height % 16 or args.width % 16:
        raise ValueError("Positive dimensions divisible by 16 are required")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "images").mkdir()
    metadata = {
        "arm": args.arm,
        "source_sha": args.source_sha,
        "client_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "endpoint": args.base_url.rstrip("/") + "/v1/images/generations",
        "batches": args.batches,
        "warmup_per_batch": args.warmup,
        "repeats_per_batch": args.repeats,
        "concurrency": 1,
        "diagnostic": args.diagnostic,
        "timing_scope": "HTTP POST through complete response read; includes server PNG/base64/transport; excludes local parsing, image decoding, checks and writes",
        "request_timeout_s": args.request_timeout,
        "b1_correctness_check": not args.skip_b1_check,
        "alternate_prompt_seed_cases": args.quality_cases,
        "artifact_policy": "Hash every decoded image; save PNG only first measured batch per shape and correctness requests; no base64 payload copies",
    }
    write_json(args.output / "metadata.json", metadata)
    rows = []
    success = False
    # Disable ambient HTTP proxies for an explicitly local service.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(batch, label, warmup, correctness_only=False, prompt=None, seed=None):
        payload = {
            "model": args.model,
            "prompt": args.prompt if prompt is None else prompt,
            "n": batch,
            "size": f"{args.width}x{args.height}",
            "num_inference_steps": args.steps,
            "guidance_scale": 1.0,
            "seed": args.seed if seed is None else seed,
            "generator_device": "cpu",
            "response_format": "b64_json",
            "output_format": "png",
            "output_compression": 100,
            "return_stage_metrics": True,
        }
        req = urllib.request.Request(
            metadata["endpoint"], data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        try:
            with opener.open(req, timeout=args.request_timeout) as response:
                status = response.status
                raw = response.read()
        except urllib.error.HTTPError as exc:
            # Preserve the actual error body, not only its status.
            raw = exc.read()
            (args.output / f"error-{label}.txt").write_bytes(raw)
            raise RuntimeError(f"HTTP {exc.code}; see error-{label}.txt") from exc
        elapsed_ms = (time.perf_counter() - start) * 1000
        body = json.loads(raw)
        if status != 200 or not isinstance(body, dict) or not isinstance(body.get("data"), list):
            write_json(args.output / f"invalid-{label}.json", body)
            raise AssertionError(f"Invalid HTTP image response: {status}")
        if len(body["data"]) != batch:
            raise AssertionError(f"Sent n={batch}, received {len(body['data'])} images")
        # Keep response metrics and transport evidence without duplicating all
        # base64 bytes on disk. Original image bytes and decoded pixel hashes
        # remain independently reviewable.
        response_metadata = {key: value for key, value in body.items() if key != "data"}
        response_metadata["data"] = [
            {key: value for key, value in item.items() if key != "b64_json"}
            for item in body["data"]
        ]
        from PIL import Image

        images = []
        for index, item in enumerate(body["data"]):
            encoded = base64.b64decode(item["b64_json"], validate=True)
            with Image.open(io.BytesIO(encoded)) as original:
                image = original.convert("RGB")
                image.load()
            if image.size != (args.width, args.height):
                raise AssertionError(f"Bad image size: {image.size}")
            save_image = correctness_only or (not warmup and label.endswith("sample-00"))
            relative = f"images/{label}-i{index}.png" if save_image else None
            if relative is not None:
                (args.output / relative).write_bytes(encoded)
            images.append({
                "file": relative, "index": index, "width": image.width, "height": image.height,
                "mode": "RGB", "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                "response_image_sha256": hashlib.sha256(encoded).hexdigest(),
                "saved_png_sha256": hashlib.sha256(encoded).hexdigest() if relative is not None else None,
            })
        row = {
            "arm": args.arm, "batch": batch, "label": label, "warmup": warmup,
            "correctness_only": correctness_only,
            "scored": not (warmup or correctness_only or args.diagnostic),
            "wall_ms": elapsed_ms, "http_status": status, "response_bytes": len(raw),
            "response_sha256": hashlib.sha256(raw).hexdigest(), "images": images,
            "server_response_fields": response_metadata,
            "server_time_metrics_ms": time_metrics_ms(body.get("metrics")),
            "payload": payload,
        }
        rows.append(row)
        with (args.output / "requests.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps({key: row[key] for key in ("arm", "batch", "label", "scored", "wall_ms", "http_status")}), flush=True)

    try:
        for batch in args.batches:
            for iteration in range(args.warmup):
                request(batch, f"b{batch}-warmup-{iteration:02}", True)
            for iteration in range(args.repeats):
                request(batch, f"b{batch}-sample-{iteration:02}", False)
        if not args.skip_b1_check and 1 not in args.batches:
            # Keep the additional batch shape out of both warmup and scores.
            request(1, "b1-correctness", False, True)
        for index in range(args.quality_cases):
            request(2, f"b2-quality-{index:02}", False, True,
                    prompt=QUALITY_PROMPTS[index], seed=args.seed + index + 1)
        summary = {}
        for batch in args.batches:
            measured = [row for row in rows if row["batch"] == batch and not row["warmup"] and not row["correctness_only"]]
            values = [row["wall_ms"] for row in measured]
            hashes = [[image["pixel_sha256"] for image in row["images"]] for row in measured]
            summary[str(batch)] = {
                "scored": not args.diagnostic, "samples_ms": values,
                "mean_ms": statistics.mean(values), "median_ms": statistics.median(values),
                "std_ms": statistics.stdev(values) if len(values) > 1 else 0,
                "min_ms": min(values), "max_ms": max(values),
                "same_seed_repeat_pixels_exact": all(value == hashes[0] for value in hashes),
                "ordered_pixel_sha256": hashes[0],
            }
            metric_keys = sorted({key for row in measured for key in row["server_time_metrics_ms"]})
            summary[str(batch)]["server_time_metrics_ms"] = {}
            for key in metric_keys:
                samples = [row["server_time_metrics_ms"][key] for row in measured if key in row["server_time_metrics_ms"]]
                summary[str(batch)]["server_time_metrics_ms"][key] = {
                    "count": len(samples), "mean_ms": statistics.mean(samples),
                    "median_ms": statistics.median(samples), "samples_ms": samples,
                }
        write_json(args.output / "summary.json", summary)
        success = True
    except BaseException as exc:
        write_json(args.output / "failure.json", {
            "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
        })
        raise
    finally:
        write_json(args.output / "validation.json", {
            "status": "PASS" if success else "FAIL", "requests_completed": len(rows),
            "scored": not args.diagnostic,
            "scope": "HTTP completion, image count/shape and within-run repeat checks; cross-arm pixel comparison is separate",
        })


if __name__ == "__main__":
    run(parse_args())
