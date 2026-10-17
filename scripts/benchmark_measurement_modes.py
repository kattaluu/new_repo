"""Compare training forward+backward cost of readout policies on identical weights (CPU)."""

import argparse
from dataclasses import replace
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if any(value < 1 for value in (args.shots, args.batch_size, args.depth, args.workers, args.repeats)):
        parser.error("All numerical options must be positive")
    import pandas as pd
    import torch
    from qml_beans.model import ModelConfig
    from qml_beans.noisy import NoiseConfig, NoisyHybridClassifier
    from qml_beans.vision_training import CPUConfig, benchmark_cpu

    torch.set_num_threads(1)
    torch.manual_seed(11)
    config = ModelConfig(196, 10, quantum_depth=args.depth, pre_layers=(32,), post_layers=(16,), shots=args.shots)
    noise = NoiseConfig()
    base = NoisyHybridClassifier(config, noise)
    x, y = torch.randn(args.batch_size, 196), torch.arange(args.batch_size) % 10
    frames = []
    for mode in ("parameter_shift", "sampled_backprop", "analytic_train"):
        model = NoisyHybridClassifier(replace(config, measurement_mode=mode), noise)
        model.load_state_dict(base.state_dict())
        frame = benchmark_cpu(model, x, y, [CPUConfig(workers=args.workers)], repeats=args.repeats)
        frame["measurement_mode"] = mode
        frames.append(frame)
    output = Path("runs/measurement_modes_benchmark")
    output.mkdir(parents=True, exist_ok=True)
    results = pd.concat(frames, ignore_index=True)
    results.to_csv(output / "timings.csv", index=False)
    metadata = dict(model_config=base.config_dict(), noise_config=base.noise_dict(), **vars(args),
                    note="Synthetic data, forward+backward, training mode, no optimizer step; warmup excluded")
    (output / "settings.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
