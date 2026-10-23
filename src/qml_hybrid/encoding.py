"""Distinct trainable encoding scales for every physical rotation occurrence."""

import torch
from torch import nn


def encoding_count(config):
    return config.quantum_depth if config.reupload else 1


def encoding_index(config, occurrence, wire):
    return 8 * occurrence + wire if config.embedding_parameterization == "per_rotation" else wire


def encoding_parameter(config):
    if not config.trainable_embedding:  # Historical checkpoints only.
        return None
    shape = (encoding_count(config), 8) if config.embedding_parameterization == "per_rotation" else (8,)
    return nn.Parameter(torch.ones(shape))


def encode_features(inputs, scale, config):
    if config.embedding_parameterization == "per_rotation":
        values = inputs.unsqueeze(-2).expand(*inputs.shape[:-1], encoding_count(config), 8)
        if scale is not None:
            values = values * scale
        return values.flatten(start_dim=-2)
    return inputs * scale if scale is not None else inputs
