"""Run a reproducible classical -> quantum-only -> joint experiment."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data")
    parser.add_argument("--target", default="Class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--classical-epochs", type=int, default=30)
    parser.add_argument("--quantum-epochs", type=int, default=10)
    parser.add_argument("--joint-epochs", type=int, default=20)
    args = parser.parse_args()
    if min(args.classical_epochs, args.quantum_epochs, args.joint_epochs) < 1:
        parser.error("All stage epoch counts must be positive")
    output = Path(args.output)
    common = [sys.executable, "-m", "qml_beans", "train", "--seed", str(args.seed)]
    if args.data:
        common += ["--data", args.data]
    stages = [
        ("classical", ["--no-quantum", "--target", args.target,
                       "--epochs", str(args.classical_epochs)]),
        ("quantum_only", ["--checkpoint", str(output / "classical" / "best.pt"),
                          "--quantum", "--no-train-classical", "--train-quantum",
                          "--epochs", str(args.quantum_epochs)]),
        ("joint", ["--checkpoint", str(output / "quantum_only" / "best.pt"),
                   "--quantum", "--train-classical", "--train-quantum",
                   "--lr", "0.0003", "--quantum-lr", "0.001", "--epochs", str(args.joint_epochs)]),
    ]
    for name, options in stages:
        subprocess.run(common + options + ["--output", str(output / name)], check=True)
    comparison = {}
    for name, _ in stages:
        metrics = json.loads((output / name / "metrics.json").read_text(encoding="utf-8"))
        comparison[name] = {"best_epoch": metrics["best_epoch"],
                            "validation_accuracy": metrics["validation"]["accuracy"],
                            "test_accuracy": metrics["test"]["accuracy"],
                            "test_macro_f1": metrics["test"]["macro_f1"],
                            "elapsed_seconds": metrics["elapsed_seconds"]}
    (output / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
