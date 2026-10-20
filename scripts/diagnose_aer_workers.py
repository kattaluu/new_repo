"""Reproduce worker startup and one forward batch without touching training runs."""
import argparse
from dataclasses import replace
import time
import torch
from qml_beans.model import ModelConfig
from qml_beans.noisy import NoiseConfig, NoisyHybridClassifier
from qml_beans.vision_training import CPUConfig, CPUEngine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=30)
    parser.add_argument('--backward', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    f = torch.load('runs/backend_benchmark_checkpoint_parameter_shift/fixture.pt', weights_only=True)
    model = NoisyHybridClassifier(ModelConfig(**f['model_config']), replace(NoiseConfig(**f['noise_config']), backend='aer_density_matrix', enabled=True, gate_errors=True))
    copies = (args.workers + 1) // 2
    x, y = f['x'].repeat(copies, 1)[:args.workers], f['y'].repeat(copies)[:args.workers]
    started = time.perf_counter()
    print('START', args.workers, flush=True)
    with CPUEngine(model, {'train': (x, y)}, CPUConfig(workers=args.workers)) as engine:
        try:
            loss, logits = engine.run('train', torch.arange(len(y)), backward=args.backward)
            if args.backward:
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            print('PASS', loss, logits.shape, time.perf_counter()-started, flush=True)
        except BaseException:
            print('WORKERS', [(p.pid, p.exitcode) for p in engine.pool._processes.values()], flush=True)
            raise

if __name__ == '__main__':
    main()
