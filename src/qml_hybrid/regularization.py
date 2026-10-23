"""Separate penalties for classical matrices, periodic gate angles and encoding scales."""

from dataclasses import dataclass
import math

from torch import nn


@dataclass(frozen=True)
class RegularizationConfig:
    classical_l2: float = 0.0
    quantum_periodic: float = 0.0
    embedding_l2: float = 0.0

    def __post_init__(self):
        for name in ('classical_l2', 'quantum_periodic', 'embedding_l2'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name}: коэффициент должен быть конечным и неотрицательным')


def regularization_penalty(model, config):
    """Only active trainable parameters contribute; biases are not penalized.

    Apply once in the parent after averaging data gradients across CPU workers.
    Frozen parameters and bypassed quantum circuits do not contribute.
    """
    penalty = next(model.parameters()).new_zeros(())
    if config.classical_l2:
        for block in model.classical_modules():
            for layer in block.modules():
                if isinstance(layer, (nn.Linear, nn.Conv2d)) and layer.weight.requires_grad:
                    penalty = penalty + .5 * config.classical_l2 * layer.weight.square().sum()
    if model.use_quantum:
        angles = model.quantum.weights
        if config.quantum_periodic and angles.requires_grad:
            penalty = penalty + config.quantum_periodic * (1 - angles.cos()).sum()
        scales = getattr(model.quantum, 'embedding_scale', None)
        if config.embedding_l2 and scales is not None and scales.requires_grad:
            penalty = penalty + .5 * config.embedding_l2 * (scales - 1).square().sum()
    return penalty
