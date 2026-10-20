"""Audit a saved vision model with fixed weights and identical validation inputs.

No training, downloads or checkpoint changes. Run from the repository root.
"""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))

import torch
from torch.nn.functional import interpolate

from qml_beans.model import ModelConfig
from qml_beans.noisy import NoiseConfig, NoisyHybridClassifier, circuit_schedule
from qml_beans.vision_data import load_raw_images
from qml_beans.vision_training import CPUConfig, CPUEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("runs/noise_audit.json"))
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = ModelConfig(**saved["model_config"])
    noise = NoiseConfig(**saved["noise_config"])
    if noise.backend != "torch_density" or not saved["mode"]["quantum"]:
        parser.error("This audit requires a torch_density checkpoint with the quantum block enabled")
    metadata = saved["data"]
    images, labels, _, _ = load_raw_images(metadata["dataset"], "data/vision", download=False)
    indices = metadata["indices"]["val"][:args.samples]
    x = torch.from_numpy(images[indices].copy()).float() / 255
    x = interpolate(x, size=(metadata["image_size"],) * 2, mode="bilinear",
                    align_corners=False, antialias=True)
    x = (2 * x - 1).flatten(1).double()
    y = torch.from_numpy(labels[indices].copy()).long()

    def make(c, n):
        model = NoisyHybridClassifier(c, replace(n, precision="float64"))
        state = {k: v for k, v in saved["state_dict"].items() if k in model.state_dict()}
        model.load_state_dict(state)
        model.set_mode(**saved["mode"])
        return model.eval()

    analytic = replace(config, shots=None)
    clean_noise = replace(noise, enabled=False, gate_errors=False)
    variants = {
        "clean": (analytic, clean_noise),
        "thermal_only": (analytic, replace(noise, gate_errors=False)),
        "gate_errors_only": (analytic, replace(noise, enabled=False)),
        "shots_only": (config, clean_noise),
        "thermal_and_gates": (analytic, noise),
        "all_noise": (config, noise),
    }
    outputs, logits, models, stats = {}, {}, {}, {}
    with torch.no_grad():
        for name, (c, n) in variants.items():
            model = models[name] = make(c, n)
            encoded = model.encoder(x)
            output = torch.cat([model.quantum(row) for row in encoded.split(1)])
            outputs[name] = output
            logits[name] = model.head(output)
            rho = model.quantum.density_matrix(encoded[:1])
            delta = output - outputs["clean"]
            stats[name] = dict(mean_abs_quantum_delta=float(delta.abs().mean()),
                               max_abs_quantum_delta=float(delta.abs().max()),
                               mean_abs_logit_delta=float((logits[name] - logits["clean"]).abs().mean()),
                               changed_predictions=int((logits[name].argmax(-1) != logits["clean"].argmax(-1)).sum()),
                               correct=int((logits[name].argmax(-1) == y).sum()),
                               first_input_purity=float((rho @ rho).diagonal(dim1=-2, dim2=-1).sum().real),
                               first_input_trace=float(rho.diagonal(dim1=-2, dim2=-1).sum().real))
            print(name, json.dumps(stats[name]), flush=True)

        # Independent PennyLane implementation, same weights and thermal/gate settings.
        reference = make(analytic, replace(noise, backend="pennylane_mixed"))
        encoded = reference.encoder(x[:2])
        reference_output = reference.quantum(encoded)
        reference_error = float((reference_output - outputs["thermal_and_gates"][:2]).abs().max())
        torch.testing.assert_close(reference_output, outputs["thermal_and_gates"][:2], atol=2e-8, rtol=2e-8)

        sampled = models["all_noise"].quantum
        sampled.set_measurement_seed(123)
        first = sampled(encoded[:1])
        sampled.set_measurement_seed(124)
        second = sampled(encoded[:1])
        seed_delta = float((first - second).abs().max())

    # Identical sharding/seeds isolate transport to spawned workers from shot randomness.
    parallel = {}
    for workers in (1, 2):
        with CPUEngine(make(config, noise), {"val": (x[:2], y[:2])},
                       CPUConfig(workers=workers, microbatch_size=1, parallel_mode="batch_queue")) as engine:
            parallel[workers] = engine.run("val", [0, 1])[1]
    torch.testing.assert_close(parallel[1], parallel[2], atol=1e-12, rtol=1e-12)
    report = dict(checkpoint=str(args.checkpoint), epoch=saved["epoch"], samples=len(x),
                  note="Fixed trained weights; first validation samples; not a retraining or accuracy comparison.",
                  model_config=saved["model_config"], noise_config=saved["noise_config"], mode=saved["mode"],
                  circuit_duration_us=sum(t for _, _, t in circuit_schedule(config, noise)),
                  variants=stats, pennylane_max_abs_error=reference_error,
                  measurement_seed_max_abs_delta=seed_delta,
                  parallel_max_abs_error=float((parallel[1] - parallel[2]).abs().max()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("variants", "model_config", "noise_config")}))


if __name__ == "__main__":
    main()
