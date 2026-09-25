#!/usr/bin/env python3
"""Run one frozen native Klein service arm; allocation/submission belongs to PJM."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request

from bench_image_service import PROMPT

MODEL_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
SOURCE_FILES = (
    "vllm_omni/diffusion/data.py",
    "vllm_omni/config/omni_config.py",
    "vllm_omni/diffusion/registry.py",
    "vllm_omni/diffusion/models/flux2/flux2_transformer.py",
    "vllm_omni/diffusion/models/flux2_klein/flux2_klein_transformer.py",
    "vllm_omni/diffusion/models/flux2_klein/pipeline_flux2_klein.py",
    "vllm_omni/diffusion/distributed/autoencoders/autoencoder_kl.py",
    "vllm_omni/diffusion/distributed/autoencoders/autoencoder_kl_flux2.py",
    "vllm_omni/diffusion/distributed/autoencoders/distributed_vae_executor.py",
    "vllm_omni/diffusion/distributed/parallel_state.py",
    "vllm_omni/diffusion/model_loader/diffusers_loader.py",
    "vllm_omni/diffusion/worker/diffusion_model_runner.py",
    "vllm_omni/entrypoints/openai/api_server.py",
    "vllm_omni/entrypoints/openai/protocol/images.py",
    "vllm_omni/entrypoints/cli/serve.py",
)


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def verify_source(root, expected, manifest_path):
    if len(expected) != 40 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("source-sha must be a full lowercase git SHA")
    if (root / ".git").exists():
        actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--"], check=True)
    else:
        actual = (root / "COMMIT").read_text().strip()
    if actual != expected:
        raise ValueError(f"Source identity {actual} != {expected}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("head_sha") != expected:
        raise ValueError("Source manifest head_sha does not match --source-sha")
    for name, expected_hash in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256(path) != expected_hash:
            raise ValueError(f"Source manifest mismatch: {name}")
    return {name: sha256(root / name) for name in sorted(set(SOURCE_FILES) | set(manifest["files"]))}


def stop_group(process, graceful_signal=signal.SIGTERM, grace=35):
    if process is None:
        return {"started": False, "group_alive": False}
    # Descendants can outlive the leader. Always address the session group.
    actions = []
    try:
        os.killpg(process.pid, graceful_signal)
        actions.append(signal.Signals(graceful_signal).name)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace
    alive = True
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            alive = False
            break
        time.sleep(0.25)
    if alive:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            actions.append("SIGKILL")
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        for _ in range(20):
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                alive = False
                break
            time.sleep(0.25)
    return {"started": True, "pid": process.pid, "returncode": process.poll(), "signals": actions, "group_alive": alive}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True, help="Node-local /tmp parent, never the Lustre artifact directory")
    parser.add_argument("--arm", choices=["default", "control", "batch"], required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[2])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--seed", type=int, default=8137)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--compile-mode", choices=["regional", "eager"], default="regional")
    parser.add_argument("--startup-timeout", type=float, default=720)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--run-timeout", type=float, default=1500)
    parser.add_argument("--skip-b1-check", action="store_true")
    parser.add_argument("--quality-cases", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--diagnostic-hook-dir", type=Path)
    parser.add_argument("--check", action="store_true", help="Validate files and print commands; do not start/import the server")
    return parser.parse_args()


def main():
    args = parse_args()
    args.source_root = args.source_root.resolve()
    args.model = args.model.resolve()
    args.output = args.output.resolve()
    args.source_manifest = args.source_manifest.resolve()
    args.cache_root = args.cache_root.resolve()
    if not args.cache_root.is_relative_to(Path("/tmp").resolve()):
        raise ValueError("--cache-root must be under node-local /tmp")
    if args.diagnostic_hook_dir and not args.diagnostic:
        raise ValueError("Diagnostic hooks must not be enabled in scored runs")
    before = verify_source(args.source_root, args.source_sha, args.source_manifest)
    manifest = json.loads(args.source_manifest.read_text())
    manifest_hash = sha256(args.source_manifest)
    if json.loads((args.model / "model_index.json").read_text()).get("_class_name") != "Flux2KleinPipeline":
        raise ValueError("Model snapshot is not native Flux2KleinPipeline")
    mode = "tile" if args.arm == "default" else "batch"
    degree = 4 if args.arm == "batch" else 1
    command = [
        sys.executable, "-m", "vllm_omni.entrypoints.cli.main", "serve", str(args.model),
        "--omni", "--host", "127.0.0.1", "--port", str(args.port),
        "--served-model-name", "klein-vae-bench", "--model-class-name", "Flux2KleinPipeline",
        "--num-gpus", "4", "--dtype", "bfloat16", "--tensor-parallel-size", "4",
        "--data-parallel-size", "1", "--pipeline-parallel-size", "1",
        "--ulysses-degree", "1", "--ring-degree", "1", "--allgather-degree", "1",
        "--cfg-parallel-size", "1", "--vae-parallel-mode", mode,
        "--vae-patch-parallel-size", str(degree), "--vae-fast-path", "off",
        "--cache-backend", "none", "--diffusion-attention-backend", "TORCH_SDPA", "--log-stats",
    ]
    if args.compile_mode == "eager":
        command.append("--enforce-eager")
    else:
        command.extend(["--diffusion-compile-granularity", "regional", "--diffusion-compile-dynamic"])
    if args.diagnostic:
        command.append("--enable-diffusion-pipeline-profiler")
    client_path = Path(__file__).with_name("bench_image_service.py")
    client_command = [
        sys.executable, str(client_path), "--base-url", f"http://127.0.0.1:{args.port}",
        "--output", str(args.output / "http"), "--arm", args.arm, "--source-sha", args.source_sha,
        "--batches", *map(str, args.batches), "--warmup", str(args.warmup), "--repeats", str(args.repeats),
        "--seed", str(args.seed), "--prompt", args.prompt, "--steps", str(args.steps),
        "--request-timeout", str(args.request_timeout),
        "--quality-cases", str(args.quality_cases),
    ]
    if args.skip_b1_check:
        client_command.append("--skip-b1-check")
    if args.diagnostic:
        client_command.append("--diagnostic")
    if args.check:
        print(json.dumps({"static_check": "PASS", "source_sha256": before,
                          "server_command": command, "client_command": client_command}, indent=2))
        return
    if "PJM_JOBID" not in os.environ:
        raise RuntimeError("Server execution requires an allocated PJM job; --check is available without one")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    # Exact source only: discard stale PYTHONPATH and earlier experiment hooks.
    paths = [str(args.source_root)]
    if args.diagnostic_hook_dir:
        paths.insert(0, str(args.diagnostic_hook_dir.resolve()))
    env["PYTHONPATH"] = os.pathsep.join(paths)
    for key in ("KV_MODEL_DIAGNOSTIC_DIR", "KV_MODEL_ARM", "KV_MODEL_RUN_ID", "VLLM_TORCH_PROFILER_DIR"):
        env.pop(key, None)
    env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               OMP_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               VLLM_WORKER_MULTIPROC_METHOD="spawn")
    for name, subdir in (("XDG_CACHE_HOME", "xdg"), ("FLASHINFER_WORKSPACE_BASE", "flashinfer"),
                         ("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
                         ("VLLM_CACHE_ROOT", "vllm"), ("CUDA_CACHE_PATH", "cuda")):
        path = args.cache_root / args.output.name / subdir
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    metadata = {
        "arm": args.arm, "source_sha": args.source_sha, "source_sha256": before,
        "source_base_sha": manifest["base_sha"], "source_patch_sha256": manifest["patch_sha256"],
        "source_manifest_sha256": manifest_hash,
        "model_revision": MODEL_REVISION, "model_path": str(args.model),
        "model_config_sha256": {name: sha256(args.model / name) for name in (
            "model_index.json", "vae/config.json", "transformer/config.json", "text_encoder/config.json")},
        "server_command": command, "client_command": client_command,
        "launcher_sha256": sha256(Path(__file__)), "client_sha256": sha256(client_path),
        "compile_mode": args.compile_mode, "diagnostic": args.diagnostic,
        "scored": not args.diagnostic, "parallelism": {"tp": 4, "sp": 1, "dp": 1, "pp": 1, "cfg": 1},
        "vae_mode": mode, "vae_degree": degree, "pjm_job_id": os.environ.get("PJM_JOBID"),
        "cache_directories": {name: env[name] for name in (
            "XDG_CACHE_HOME", "FLASHINFER_WORKSPACE_BASE", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR")},
        "versions": {}, "started_unix_s": time.time(), "status": "STARTING",
    }
    for package in ("torch", "vllm", "vllm-omni", "diffusers", "transformers", "Pillow"):
        try:
            metadata["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            metadata["versions"][package] = "not installed as a distribution"
    write_json(args.output / "metadata.json", metadata)
    server = client = None
    success = False
    run_start = time.monotonic()

    def interrupted(signum, frame):
        raise InterruptedError(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with (args.output / "server.log").open("wb") as server_log, (args.output / "client.log").open("wb") as client_log:
            server = subprocess.Popen(command, cwd=args.source_root, env=env, stdout=server_log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            metadata["server_pid"] = server.pid
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited before readiness: {server.returncode}")
                if time.monotonic() - run_start > min(args.startup_timeout, args.run_timeout):
                    raise TimeoutError("Server readiness timed out; see server.log")
                try:
                    with opener.open(f"http://127.0.0.1:{args.port}/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(1)
            metadata["health_ready_ms"] = (time.monotonic() - run_start) * 1000
            metadata["status"] = "RUNNING"
            write_json(args.output / "metadata.json", metadata)
            client = subprocess.Popen(client_command, cwd=args.output, env=env, stdout=client_log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            while client.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited during benchmark: {server.returncode}")
                if time.monotonic() - run_start > args.run_timeout:
                    raise TimeoutError("Whole service-arm budget expired")
                time.sleep(1)
            if client.returncode != 0:
                raise RuntimeError(f"HTTP client failed: {client.returncode}; see client.log")
            success = True
    except BaseException as exc:
        write_json(args.output / "failure.json", {
            "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(),
        })
        raise
    finally:
        # Ignore a second termination signal only while cleaning up our own
        # tracked process groups; PJM's outer hard deadline remains effective.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        client_cleanup = stop_group(client, grace=5)
        server_cleanup = stop_group(server, graceful_signal=signal.SIGINT)
        try:
            source_unchanged = (
                verify_source(args.source_root, args.source_sha, args.source_manifest) == before
                and sha256(args.source_manifest) == manifest_hash
            )
        except Exception as exc:
            source_unchanged = False
            metadata["source_verification_error"] = str(exc)
        cleanup_complete = not client_cleanup["group_alive"] and not server_cleanup["group_alive"]
        metadata.update(status="PASS" if success and source_unchanged and cleanup_complete else "FAIL",
                        ended_unix_s=time.time(), source_unchanged=source_unchanged,
                        cleanup_complete=cleanup_complete, client_cleanup=client_cleanup,
                        server_cleanup=server_cleanup)
        write_json(args.output / "metadata.json", metadata)
        write_json(args.output / "validation.json", {
            key: metadata[key] for key in ("status", "scored", "diagnostic", "source_unchanged", "cleanup_complete")
        })
    if metadata["status"] != "PASS":
        raise RuntimeError("Source/cleanup qualification failed")
    print("MODEL_SERVICE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
