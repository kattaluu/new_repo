"""Compare warmed forward/backward times with fixed gate errors off/on.

Run from the project root. No optimizer updates; same starting weights and inputs.
Initialization/compilation is timed separately and excluded from training timings.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time

import pandas as pd
import torch

from qml_beans.model import ModelConfig
from qml_beans.noisy import NoiseConfig, NoisyHybridClassifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['torch_density', 'pennylane_mixed', 'pennylane_statevector'], default='torch_density')
    parser.add_argument('--topology', choices=['linear', 'brickwork', 'brickwork3', 'ring'], default='brickwork')
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--t1t2', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('runs/benchmark_gate_errors'))
    args = parser.parse_args()
    if min(args.depth, args.batch_size, args.repeats, args.threads) < 1:
        parser.error('Depth, batch size, repeats and threads must be positive')
    if args.t1t2 and args.backend == 'pennylane_statevector':
        parser.error('Statevector does not support T1/T2')
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    config = ModelConfig(16, 10, quantum_depth=args.depth, entanglement=args.topology)
    noise = NoiseConfig(backend=args.backend, enabled=args.t1t2)
    models, compilation = {}, {}
    for enabled in (False, True):
        started = time.perf_counter()
        models[enabled] = NoisyHybridClassifier(config, replace(noise, gate_errors=enabled))
        compilation[enabled] = time.perf_counter() - started
    models[True].load_state_dict(models[False].state_dict())
    x, y = torch.randn(args.batch_size, 16), torch.arange(args.batch_size) % 10

    def step(model):
        model.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(x), y).backward()

    for model in models.values():
        step(model)
        step(model)
    timings = {False: [], True: []}
    for repeat in range(args.repeats):
        for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
            started = time.perf_counter()
            step(models[enabled])
            timings[enabled].append(time.perf_counter() - started)
    records = [dict(gate_errors=enabled, median_seconds=statistics.median(values),
                    min_seconds=min(values), initialization_seconds=compilation[enabled],
                    repeats=len(values), seconds=values) for enabled, values in timings.items()]
    ratio = records[1]['median_seconds'] / records[0]['median_seconds']
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'benchmark.json').write_text(json.dumps(dict(
        settings={**vars(args), 'output': str(args.output)}, results=records,
        time_ratio=ratio, torch_version=torch.__version__), indent=2), encoding='utf-8')
    table = pd.DataFrame([{k: v for k, v in row.items() if k != 'seconds'} for row in records])
    table.to_csv(args.output/'benchmark.csv', index=False)
    print(table.to_string(index=False))
    print(f'Time ratio (errors on/off): {ratio:.3f}')


if __name__ == '__main__':
    main()
