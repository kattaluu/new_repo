"""Ghost BatchNorm statistics for synchronous CPU microbatch training."""

from contextlib import contextmanager

import torch
from torch import nn


def merge_moments(first, second):
    """Combine (count, mean, centered sum of squares) without cancellation."""
    if first is None:
        return second
    n, mean, m2 = first
    count, other, other_m2 = second
    delta = other - mean
    total = n + count
    return total, mean + delta * (count / total), m2 + other_m2 + delta.square() * (n * count / total)


@contextmanager
def collect_batchnorm_moments(model):
    statistics, handles = {}, []

    def hook(name):
        def collect(module, inputs):
            if not module.training or not module.track_running_stats:
                return
            value = inputs[0].detach()
            count = value.numel() // value.shape[1]
            # Singleton forwards use existing running statistics, not batch statistics.
            if count <= 1:
                return
            dims = (0, *range(2, value.ndim))
            variance, mean = torch.var_mean(value.to(torch.float64), dim=dims, unbiased=False)
            statistics[name] = merge_moments(statistics.get(name), (count, mean, variance * count))
        return collect

    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training:
            handles.append(module.register_forward_pre_hook(hook(name)))
    try:
        yield statistics
    finally:
        for handle in handles:
            handle.remove()


def snapshot_batchnorm(model):
    return {name: (module.running_mean.clone(), module.running_var.clone(), int(module.num_batches_tracked))
            for name, module in model.named_modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.track_running_stats and module.training}


def update_batchnorm(model, previous, results):
    """One running-stat update per logical batch; workers never mutate parent buffers.

    Forward normalization and gradients use each microbatch's local statistics.
    Evaluation uses pooled moments of observed activations, including between-shard variance.
    """
    modules = dict(model.named_modules())
    with torch.no_grad():
        for name, (old_mean, old_var, batches) in previous.items():
            combined = None
            for stats in results:
                if name in stats:
                    combined = merge_moments(combined, stats[name])
            module = modules[name]
            module.running_mean.copy_(old_mean)
            module.running_var.copy_(old_var)
            module.num_batches_tracked.fill_(batches)
            if combined is None:
                continue
            count, mean, m2 = combined
            module.num_batches_tracked.add_(1)
            momentum = module.momentum if module.momentum is not None else 1 / (batches + 1)
            module.running_mean.lerp_(mean.to(module.running_mean), momentum)
            module.running_var.lerp_((m2 / (count - 1)).to(module.running_var), momentum)
