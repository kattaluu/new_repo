"""Serializable PyTorch classical blocks, reconstructible in CPU workers."""

import json
import math

import torch
from torch import nn


_FIELDS = {
    "Linear": ("in_features", "out_features", "bias"),
    "Conv2d": ("in_channels", "out_channels", "kernel_size", "stride", "padding",
               "dilation", "groups", "bias", "padding_mode"),
    "BatchNorm1d": ("num_features", "eps", "momentum", "affine", "track_running_stats"),
    "BatchNorm2d": ("num_features", "eps", "momentum", "affine", "track_running_stats"),
    "MaxPool2d": ("kernel_size", "stride", "padding", "dilation", "return_indices", "ceil_mode"),
    "AvgPool2d": ("kernel_size", "stride", "padding", "ceil_mode", "count_include_pad", "divisor_override"),
    "AdaptiveAvgPool2d": ("output_size",),
    "Flatten": ("start_dim", "end_dim"),
    "Unflatten": ("dim", "unflattened_size"),
    "ReLU": ("inplace",), "LeakyReLU": ("negative_slope", "inplace"),
    "ELU": ("alpha", "inplace"), "GELU": ("approximate",), "SiLU": ("inplace",),
    "Tanh": (), "Sigmoid": (), "Identity": (),
}


class _SingletonBatchNorm:
    """Use existing running statistics when a channel has only one value."""
    def forward(self, inputs):
        self._check_input_dim(inputs)
        count = inputs.numel() // inputs.shape[1]
        if count == 1 and (self.training or not self.track_running_stats):
            if not self.track_running_stats:
                raise ValueError("BatchNorm: при одном значении на канал нужен track_running_stats=True")
            return torch.nn.functional.batch_norm(inputs, self.running_mean, self.running_var,
                                                  self.weight, self.bias, False, 0., self.eps)
        return super().forward(inputs)


class BatchNorm1d(_SingletonBatchNorm, nn.BatchNorm1d):
    pass


class BatchNorm2d(_SingletonBatchNorm, nn.BatchNorm2d):
    pass


def normalize_spec(spec):
    """JSON-compatible architecture only: no weights, executable code or pickle."""
    if spec is None:
        return None
    if not isinstance(spec, (list, tuple)):
        raise ValueError("Архитектура должна быть списком описаний слоёв")
    result = []
    for layer in spec:
        if not isinstance(layer, dict) or layer.get("type") not in (*_FIELDS, "Sequential"):
            raise ValueError(f"Неподдерживаемое описание слоя: {layer!r}")
        name = layer["type"]
        if name == "Sequential":
            if set(layer) != {"type", "layers"}:
                raise ValueError("Sequential: ожидаются поля type и layers")
            result.append(dict(type=name, layers=list(normalize_spec(layer["layers"]))))
        else:
            if set(layer) - {"type", *_FIELDS[name]}:
                raise ValueError(f"Неизвестные параметры слоя {name}")
            result.append(json.loads(json.dumps(layer, allow_nan=False)))
    return tuple(result)


def sequential_spec(module):
    """Describe a Sequential without saving its initialized weights."""
    if type(module) is not nn.Sequential:
        raise ValueError("Задайте блок через torch.nn.Sequential")
    layers = []
    for child in module.children():
        name = type(child).__name__
        if type(child) is nn.Sequential:
            layers.append(dict(type="Sequential", layers=list(sequential_spec(child))))
        elif name in _FIELDS and (type(child) is getattr(nn, name) or type(child) in (BatchNorm1d, BatchNorm2d)):
            options = {field: (getattr(child, field) is not None if field == "bias" else getattr(child, field))
                       for field in _FIELDS[name]}
            if name == "MaxPool2d" and options["return_indices"]:
                raise ValueError("MaxPool2d должен возвращать тензор: return_indices=False")
            layers.append(dict(type=name, **options))
        else:
            raise ValueError(f"Слой {name} пока не поддерживает переносимую конфигурацию. "
                             f"Поддерживаются: {', '.join(_FIELDS)} и вложенный Sequential")
    return normalize_spec(layers)


def sequential_from_spec(spec):
    modules = []
    for layer in normalize_spec(spec):
        options = {key: value for key, value in layer.items() if key != "type"}
        name = layer["type"]
        try:
            module = (sequential_from_spec(options["layers"]) if name == "Sequential"
                      else {"BatchNorm1d": BatchNorm1d, "BatchNorm2d": BatchNorm2d}.get(name, getattr(nn, name))(**options))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Некорректные параметры слоя {name}: {exc}") from exc
        if isinstance(module, nn.MaxPool2d) and module.return_indices:
            raise ValueError("MaxPool2d: нужен return_indices=False")
        modules.append(module)
    return nn.Sequential(*modules)


def classical_architecture(encoder, head=None, *, image_shape=None, latent_features=8):
    """Return ModelConfig kwargs (encoder -> latent_features, head -> classes).

    With image_shape=(C,H,W), flattened features are reshaped before the encoder.
    Modules specify structure only. Their weights are initialized anew in each model.
    Widths other than 8 enable model-owned adapters around the eight-qubit block.
    """
    result = dict(encoder_spec=sequential_spec(encoder),
                head_spec=sequential_spec(head) if head is not None else None,
                image_shape=tuple(image_shape) if image_shape is not None else None)
    if latent_features != 8:
        result["latent_features"] = latent_features
    return result


def build_classical_block(spec, n_inputs, n_outputs, *, image_shape=None, label="Блок"):
    block = sequential_from_spec(spec)
    if image_shape is not None:
        block = nn.Sequential(nn.Unflatten(1, tuple(image_shape)), *block.children())
    try:
        # Shape validation must not update BatchNorm running statistics.
        block.eval()
        with torch.no_grad():
            output = block(torch.zeros(2, n_inputs))
        block.train()
    except (RuntimeError, ValueError, TypeError, IndexError) as exc:
        raise ValueError(f"{label}: размеры слоёв несовместимы с входом: {exc}") from exc
    if not isinstance(output, torch.Tensor) or output.shape != (2, n_outputs):
        raise ValueError(f"{label}: нужен выход (batch, {n_outputs}), "
                         f"получен {getattr(output, 'shape', type(output))}")
    return block


def normalize_image_shape(shape, n_features):
    if shape is None:
        return None
    shape = tuple(shape)
    if len(shape) != 3 or any(type(x) is not int or x < 1 for x in shape) or math.prod(shape) != n_features:
        raise ValueError("image_shape: нужны (каналы, высота, ширина) с произведением n_features")
    return shape
