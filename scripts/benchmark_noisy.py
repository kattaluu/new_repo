"""Small reproducible CPU benchmark, not an accuracy study."""

import json
import os
from pathlib import Path


def main():
    import torch
    from qml_beans.model import ModelConfig
    from qml_beans.noisy import NoisyHybridClassifier, NoiseConfig
    from qml_beans.vision_training import benchmark_cpu, CPUConfig

    torch.set_num_threads(1)
    torch.manual_seed(11)
    config = ModelConfig(196, 10, pre_layers=(32,), post_layers=(16,), quantum_depth=1, entanglement="ring")
    base = NoisyHybridClassifier(config, NoiseConfig())
    x, y = torch.randn(8, 196), torch.arange(8) % 10
    results = []
    for backend in ("torch_density", "pennylane_mixed"):
        model = NoisyHybridClassifier(config, NoiseConfig(backend=backend))
        model.load_state_dict(base.state_dict(), strict=False)
        results.append(benchmark_cpu(model, x, y, [CPUConfig(workers=1), CPUConfig(workers=2)], repeats=3))
    import pandas as pd
    frame = pd.concat(results, ignore_index=True)
    output = Path("runs/noisy_backend_benchmark.csv")
    frame.to_csv(output, index=False)
    import pennylane
    output.with_suffix(".json").write_text(json.dumps(dict(
        model=config.__dict__, noise=base.noise_dict(), torch=str(torch.__version__),
        pennylane=pennylane.__version__, cpu=os.environ.get("PROCESSOR_IDENTIFIER"),
        logical_cpus=os.cpu_count(), batch_size=8, repeats=3, initialization_excluded=True,
        optimizer_step=False, input="synthetic, fixed seed 11"), ensure_ascii=False, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
