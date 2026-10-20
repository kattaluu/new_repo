"""Compare complete training steps using fixed inputs/weights, in isolated CPU processes.

Run from the repo root. This does not interrupt other training or modify checkpoints.
Results explicitly retain concurrent CPU load, raw repeats and timeout/error status.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time


BACKENDS = ("pennylane_statevector", "aer_statevector", "torch_density", "pennylane_mixed", "aer_density_matrix")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def run_case(args):
    import numpy as np
    import psutil
    import torch
    from threadpoolctl import threadpool_limits
    from qml_beans.model import ModelConfig
    from qml_beans.noisy import NoiseConfig, NoisyHybridClassifier

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    limits = threadpool_limits(limits=1)
    fixture = torch.load(args.output / "fixture.pt", map_location="cpu", weights_only=True)
    case = json.loads(args.case)
    config = ModelConfig(**fixture["model_config"])
    config = replace(config, shots=None if case["mode"] == "analytic" else (config.shots or 1000),
                     measurement_mode="parameter_shift" if case["mode"] == "analytic" else case["mode"])
    noise = NoiseConfig(**fixture["noise_config"])
    noise = replace(noise, backend=case["backend"], enabled=case["noise"] == "thermal_gates",
                    gate_errors=case["noise"] != "clean")
    started = time.perf_counter()
    model = NoisyHybridClassifier(config, noise)
    model.load_state_dict({k: v for k, v in fixture["weights"].items() if k in model.state_dict()})
    init_seconds = time.perf_counter() - started
    x, y = fixture["x"], fixture["y"]
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)

    def reset():
        model.load_state_dict(initial)
        optimizer.state.clear()
        # Allocate Adam state before timing, avoiding first-step setup bias.
        model.zero_grad(set_to_none=True)
        for p in model.parameters():
            p.grad = torch.zeros_like(p)
        optimizer.step()
        model.load_state_dict(initial)
        model.zero_grad(set_to_none=True)

    record = dict(**case, status="running", init_seconds=init_seconds, measurements=[],
                  model_config=asdict(config), noise_config=asdict(noise))
    path = args.output / (case["id"] + ".json")
    write_json(path, record)
    for repeat in range(-1, args.repeats):
        reset()
        model.train()
        model.quantum.set_measurement_seed(100 + repeat)
        system_start = psutil.cpu_times()
        wall, cpu = time.perf_counter(), time.process_time()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        forward = time.perf_counter()
        loss.backward()
        backward = time.perf_counter()
        optimizer.step()
        end, cpu_end = time.perf_counter(), time.process_time()
        system_end = psutil.cpu_times()
        total = sum(system_end) - sum(system_start)
        idle = system_end.idle - system_start.idle
        if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError("Non-finite loss or missing/non-finite gradient")
        row = dict(repeat=repeat, seconds=end-wall, cpu_seconds=cpu_end-cpu,
                   forward_seconds=forward-wall, backward_seconds=backward-forward,
                   optimizer_seconds=end-backward, cpu_load_percent=100 * (1-idle/total) if total else None,
                   loss=float(loss.detach()))
        if repeat == -1:
            record["warmup"] = row
        else:
            record["measurements"].append(row)
        write_json(path, record)
        print(case["id"], "warmup" if repeat == -1 else repeat, f'{row["seconds"]:.4f}s', flush=True)
    # Evaluation forward pass: analytic_train evaluates with shots, so this is
    # the relevant finite-shot evaluation timing, not an extra training policy.
    model.eval()
    evaluation = []
    with torch.no_grad():
        for repeat in range(args.repeats):
            wall = time.perf_counter()
            model(x)
            evaluation.append(time.perf_counter()-wall)
    record.update(status="ok", eval_seconds=evaluation)
    write_json(path, record)
    del limits


def prepare(args):
    import torch
    import psutil
    from torch.nn.functional import interpolate
    from qml_beans.model import ModelConfig
    from qml_beans.noisy import NoiseConfig, NoisyHybridClassifier
    from qml_beans.vision_data import load_raw_images

    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    checkpoint_bytes = args.checkpoint.read_bytes()
    saved = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    config = ModelConfig(**saved["model_config"])
    noise = replace(NoiseConfig(**saved["noise_config"]), backend="torch_density", enabled=False, gate_errors=False)
    if args.depth is not None:
        config = replace(config, quantum_depth=args.depth)
    metadata = saved["data"]
    raw, labels, _, _ = load_raw_images(metadata["dataset"], "data/vision", download=False)
    indices = metadata["indices"]["train"][:args.batch_size]
    x = torch.from_numpy(raw[indices].copy()).float()/255
    x = (2*interpolate(x, size=(metadata["image_size"],)*2, mode="bilinear", align_corners=False, antialias=True)-1).flatten(1)
    y = torch.from_numpy(labels[indices].copy()).long()
    torch.manual_seed(71)
    model = NoisyHybridClassifier(config, noise)
    if args.depth is None:
        model.load_state_dict(saved["state_dict"], strict=False)
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    torch.save(dict(model_config=asdict(config), noise_config=asdict(noise), weights=weights, x=x, y=y), args.output/"fixture.pt")
    write_json(args.output/"settings.json", dict(checkpoint=str(args.checkpoint), model_config=asdict(config),
        noise_config=asdict(noise), batch_size=len(x), repeats=args.repeats, workers=1, threads=1,
        checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(), checkpoint_epoch=saved.get("epoch"),
        checkpoint_mode=saved["mode"], checkpoint_settings=saved["settings"],
        initialization="checkpoint weights" if args.depth is None else "random seed 71 (depth override)",
        train_indices=indices, input_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),
        versions={p: importlib.metadata.version(p) for p in ("torch", "pennylane", "qiskit", "qiskit-aer", "numpy")},
        cpu=os.environ.get("PROCESSOR_IDENTIFIER"), physical_cpus=psutil.cpu_count(logical=False),
        logical_cpus=os.cpu_count(), cpu_load_before=psutil.cpu_percent(interval=1),
        limitations="Concurrent user training; one process/one thread; microbatch timing, not full-epoch or workers scaling. Shared precision policy; backend internals can differ.",
        timing="Forward + cross-entropy + backward + Adam step. One warmup. Same weights restored before each repeat. Initialization/reset excluded."))


def summarize(output, selected_ids=None):
    import csv
    rows = []
    for path in sorted(output.glob("case_*.json")):
        if selected_ids is not None and path.stem not in selected_ids:
            continue
        result = json.loads(path.read_text())
        samples = result.get("measurements", [])
        row = {k: result[k] for k in ("id", "backend", "noise", "mode", "status")}
        row["repeats"] = len(samples)
        if samples:
            row.update(median_seconds=statistics.median(s["seconds"] for s in samples),
                       min_seconds=min(s["seconds"] for s in samples), max_seconds=max(s["seconds"] for s in samples),
                       median_cpu_seconds=statistics.median(s["cpu_seconds"] for s in samples),
                       median_cpu_load=statistics.median(s["cpu_load_percent"] for s in samples))
        rows.append(row)
    fields = ["id", "backend", "noise", "mode", "status", "repeats", "median_seconds", "min_seconds", "max_seconds", "median_cpu_seconds", "median_cpu_load"]
    with (output/"summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/backend_benchmark"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--case")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--modes", nargs="+", choices=("analytic", "sampled_backprop", "parameter_shift"),
                        default=["parameter_shift"])
    parser.add_argument("--noise", nargs="+", choices=("clean", "gates", "thermal_gates"),
                        default=["clean", "thermal_gates"])
    parser.add_argument("--matched-representations", action="store_true",
                        help="Compare clean statevectors and noisy density matrices only")
    args = parser.parse_args()
    if min(args.batch_size, args.repeats, args.timeout) < 1:
        parser.error("Sizes, repeats and timeout must be positive")
    if args.case:
        run_case(args)
        return
    if not args.resume:
        if (args.output/"fixture.pt").exists():
            parser.error("Output exists; use --resume or another output directory")
        if args.checkpoint is None:
            parser.error("--checkpoint is required")
        prepare(args)
    cases = [dict(backend=b, noise=n, mode=m, id=f"case_{n}_{m}_{b}")
             for n in args.noise
             for m in args.modes for b in BACKENDS
             if (n != "thermal_gates" or "statevector" not in b)
             and (not args.matched_representations or
                  (n == "clean" and "statevector" in b) or
                  (n == "thermal_gates" and "statevector" not in b))]
    random.Random(71).shuffle(cases)
    selected_ids = {case["id"] for case in cases}
    write_json(args.output/"comparison_plan.json", dict(cases=cases, repeats=args.repeats,
               matched_representations=args.matched_representations, timeout=args.timeout))
    summarize(args.output, selected_ids)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MPLCONFIGDIR=str(Path(".cache/matplotlib").resolve()))
    for case in cases:
        path = args.output/(case["id"]+".json")
        if args.resume and path.exists() and json.loads(path.read_text()).get("status") == "ok":
            continue
        print("START", case["id"], flush=True)
        cmd = [sys.executable, __file__, "--case", json.dumps(case), "--output", str(args.output), "--repeats", str(args.repeats)]
        with (args.output/(case["id"]+".log")).open("w", encoding="utf-8") as log:
            process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                status = "ok" if process.wait(timeout=args.timeout) == 0 else "error"
            except subprocess.TimeoutExpired:
                # Windows venv launchers have a child interpreter. Stop the whole
                # owned tree before the next case, not just its launcher.
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                else:
                    process.kill()
                process.wait()
                status = "timeout"
        record = json.loads(path.read_text()) if path.exists() else dict(**case, measurements=[])
        record["status"] = status
        write_json(path, record)
        rows = summarize(args.output, selected_ids)
        latest = next(r for r in rows if r["id"] == case["id"])
        print("DONE", json.dumps(latest), flush=True)


if __name__ == "__main__":
    main()
