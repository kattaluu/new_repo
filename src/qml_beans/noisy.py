"""Exact eight-qubit density matrices and a PennyLane reference with identical timing."""

from dataclasses import asdict, dataclass
import math

import pennylane as qml
import numpy as np
import torch
from torch import nn

from .model import HybridClassifier, ModelConfig, QuantumBlock
from .measurements import (effective_shots, readout_with_analytic_jacobian, rotate_measurement_basis,
                           sample_expectations, sampled_with_analytic_jacobian)


@dataclass(frozen=True)
class NoiseConfig:
    # All times are microseconds; tuple entries follow wire order 0..7.
    t1_us: tuple[float, ...] = (100.0,) * 8
    t2_us: tuple[float, ...] = (80.0,) * 8
    single_gate_us: float = 0.05
    two_gate_us: float = 0.3
    readout_idle_us: float = 0.0
    enabled: bool = True
    backend: str = "torch_density"
    precision: str = "float32"

    def __post_init__(self):
        for name in ("t1_us", "t2_us"):
            values = getattr(self, name)
            if isinstance(values, (int, float)):
                values = (float(values),) * 8
            values = tuple(float(value) for value in values)
            object.__setattr__(self, name, values)
            if len(values) != 8 or any(not math.isfinite(t) or t <= 0 for t in values):
                raise ValueError(f"{name}: нужны восемь конечных положительных времён")
        if any(t2 > 2 * t1 for t1, t2 in zip(self.t1_us, self.t2_us)):
            raise ValueError("Физическое ограничение: T2 <= 2*T1 для каждого кубита")
        for name in ("single_gate_us", "two_gate_us", "readout_idle_us"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} должно быть конечным и неотрицательным")
        if self.backend not in ("torch_density", "pennylane_mixed", "pennylane_statevector"):
            raise ValueError("Неизвестный шумовой бэкенд")
        if self.backend == "pennylane_statevector" and self.enabled:
            raise ValueError("Идеальный state vector не поддерживает T1/T2: отключите шум или выберите матрицу плотности")
        if self.precision not in ("float32", "float64"):
            raise ValueError("precision: float32 или float64")


def circuit_schedule(config, noise):
    """Parallel one-qubit moments; serial two-qubit gates; idles on ALL wires.

    Noise is applied after each ideal moment (gate-level, zero-temperature Markov model).
    A virtual RZ is not assumed: every configured rotation has single_gate_us duration.
    """
    operations = []
    if config.initial_hadamard:
        operations.append(("H", None, noise.single_gate_us))
    operations.append(("embed", None, noise.single_gate_us))
    for layer in range(config.quantum_depth):
        if layer and config.reupload:
            operations.append(("embed", None, noise.single_gate_us))
        for gate_index, gate in enumerate(config.rotation_gates):
            operations.append((gate, (layer, gate_index), noise.single_gate_us))
        for edge in config.edges():
            operations.append((config.entangler, edge, noise.two_gate_us))
    operations.append(("idle", None, noise.readout_idle_us))
    return operations


def _rotation(gate, theta):
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    z = torch.zeros_like(c)
    if gate == "RX":
        values = (c + 0j, -1j * s, -1j * s, c + 0j)
    elif gate == "RY":
        values = (c + 0j, -s + 0j, s + 0j, c + 0j)
    elif gate == "RZ":
        values = (torch.exp(-0.5j * theta), z + 0j, z + 0j, torch.exp(0.5j * theta))
    else:
        raise ValueError(gate)
    return torch.stack(values, dim=-1).reshape(-1, 2, 2)


def apply_unitary(rho, unitary, wire, n_qubits=8):
    batch, size, _ = rho.shape
    left, right = 2**wire, 2 ** (n_qubits - wire - 1)
    unitary = unitary.reshape(-1, 2, 2)

    def rows(matrix, gate):
        shape = matrix.reshape(batch, left, 2, right, size)
        return torch.einsum("bij,bpjqk->bpiqk", gate, shape).reshape(batch, size, size)

    result = rows(rho, unitary)
    return rows(result.transpose(-1, -2), unitary.conj()).transpose(-1, -2)


def thermal_channel(rho, wire, duration, t1, t2, n_qubits=8):
    """rho11 decays by exp(-t/T1); coherences by exp(-t/T2), including T1 contribution."""
    if duration == 0:
        return rho
    shape = [rho.shape[0]] + [2] * (2 * n_qubits)
    axes = [0, wire + 1, wire + n_qubits + 1]
    axes += [axis for axis in range(1, 2 * n_qubits + 1) if axis not in axes]
    inverse = [axes.index(axis) for axis in range(len(axes))]
    blocks = rho.reshape(shape).permute(axes).reshape(rho.shape[0], 2, 2, -1)
    e1, e2 = math.exp(-duration / t1), math.exp(-duration / t2)
    result = torch.stack((blocks[:, 0, 0] + (1 - e1) * blocks[:, 1, 1],
                          e2 * blocks[:, 0, 1], e2 * blocks[:, 1, 0],
                          e1 * blocks[:, 1, 1]), dim=1)
    return result.reshape([rho.shape[0]] + [2] * (2 * n_qubits)).permute(inverse).reshape_as(rho)


class TorchDensityBlock(nn.Module):
    def __init__(self, config, noise):
        super().__init__()
        self.config, self.noise = config, noise
        self.weights = nn.Parameter(0.05 * torch.randn(config.quantum_depth, 8, len(config.rotation_gates)))
        self.schedule = circuit_schedule(config, noise)
        self.sampling_rng = torch.Generator().manual_seed(config.measurement_seed)
        indices = torch.arange(256)
        self.register_buffer("z_signs", torch.stack([1 - 2 * ((indices >> (7 - wire)) & 1)
                                                    for wire in range(8)], dim=1).float())

    def set_measurement_seed(self, seed):
        self.sampling_rng.manual_seed(seed)

    def density_matrix(self, inputs, weights=None, shift=None):
        weights = self.weights if weights is None else weights
        dtype = torch.complex128 if inputs.dtype == torch.float64 else torch.complex64
        ket = torch.zeros((inputs.shape[0], 256), dtype=dtype, device=inputs.device)
        ket[:, 0] = 1
        rho = ket.unsqueeze(-1) * ket.conj().unsqueeze(-2)
        indices = torch.arange(256, device=inputs.device)
        for moment, (gate, info, duration) in enumerate(self.schedule):
            if gate in ("CNOT", "CZ"):
                control, target = info
                active = ((indices >> (7 - control)) & 1).bool()
                if gate == "CNOT":
                    permutation = indices ^ (active.long() << (7 - target))
                    rho = rho[:, permutation][:, :, permutation]
                else:
                    signs = 1 - 2 * (active & ((indices >> (7 - target)) & 1).bool()).to(inputs.dtype)
                    rho = rho * signs[None, :, None] * signs[None, None, :]
            elif gate != "idle":
                for wire in range(8):
                    if gate == "H":
                        unitary = inputs.new_tensor([[1, 1], [1, -1]]).to(dtype) / math.sqrt(2)
                    elif gate == "embed":
                        theta = torch.pi * inputs[:, wire]
                        if shift is not None and (moment, wire) == shift[:2]:
                            theta = theta + shift[2]
                        unitary = _rotation(self.config.embedding, theta)
                    else:
                        layer, gate_index = info
                        theta = weights[layer, wire, gate_index]
                        if shift is not None and (moment, wire) == shift[:2]:
                            theta = theta + shift[2]
                        unitary = _rotation(gate, theta)
                    rho = apply_unitary(rho, unitary, wire)
            if self.noise.enabled and duration:
                for wire in range(8):
                    rho = thermal_channel(rho, wire, duration, self.noise.t1_us[wire], self.noise.t2_us[wire])
        return rho

    def _measure(self, inputs, weights=None, shift=None):
        rho = self.density_matrix(inputs, weights, shift)
        if self.config.measurement == "X":
            h = inputs.new_tensor([[1, 1], [1, -1]]).to(rho.dtype) / math.sqrt(2)
            for wire in range(8):
                rho = apply_unitary(rho, h, wire)
        elif self.config.measurement == "Y":
            # H S^dagger rotates Y measurement into Z, no physical gates/timing added.
            u = inputs.new_tensor([[1, 1], [1, -1]]).to(rho.dtype) / math.sqrt(2)
            u = u @ torch.diag(torch.tensor([1, -1j], dtype=rho.dtype, device=rho.device))
            for wire in range(8):
                rho = apply_unitary(rho, u, wire)
        probabilities = rho.diagonal(dim1=-2, dim2=-1).real
        mean = probabilities @ self.z_signs.to(inputs.dtype)
        shots = effective_shots(self.config, self.training)
        if shots:
            with torch.no_grad():
                sampled = sample_expectations(probabilities.detach(), shots, self.sampling_rng)
            if self.config.measurement_mode != "parameter_shift":
                return sampled_with_analytic_jacobian(mean, sampled)
            return sampled
        return mean

    def forward(self, inputs):
        if self.config.shots and self.config.measurement_mode == "parameter_shift":
            return _ShotParameterShift.apply(inputs, self.weights, self)
        return self._measure(inputs)


class _ShotParameterShift(torch.autograd.Function):
    """Finite-shot forward AND finite-shot gate-wise parameter-shift, without analytic surrogate gradients."""

    @staticmethod
    def forward(ctx, inputs, weights, block):
        ctx.block = block
        ctx.save_for_backward(inputs, weights)
        return block._measure(inputs, weights)

    @staticmethod
    def backward(ctx, upstream):
        inputs, weights = ctx.saved_tensors
        block = ctx.block
        input_gradient = torch.zeros_like(inputs) if ctx.needs_input_grad[0] else None
        weight_gradient = torch.zeros_like(weights) if ctx.needs_input_grad[1] else None
        with torch.no_grad():
            for moment, (gate, info, _) in enumerate(block.schedule):
                is_input = gate == "embed"
                if (is_input and input_gradient is None) or (not is_input and weight_gradient is None):
                    continue
                if not is_input and gate not in ("RX", "RY", "RZ"):
                    continue
                for wire in range(8):
                    plus = block._measure(inputs, weights, (moment, wire, math.pi / 2))
                    minus = block._measure(inputs, weights, (moment, wire, -math.pi / 2))
                    contribution = ((plus - minus) * upstream).sum(dim=-1) / 2
                    if is_input:
                        # Reuploaded features have multiple gate occurrences; sum their derivatives.
                        input_gradient[:, wire] += math.pi * contribution
                    else:
                        layer, gate_index = info
                        weight_gradient[layer, wire, gate_index] += contribution.sum()
        return input_gradient, weight_gradient, None


class PennyLaneMixedBlock(nn.Module):
    def __init__(self, config, noise):
        super().__init__()
        self.config = config
        self.shots = config.shots
        self.sampling_rng = np.random.default_rng(config.measurement_seed)
        self.backprop_sampling_rng = torch.Generator().manual_seed(config.measurement_seed)
        self.weights = nn.Parameter(0.05 * torch.randn(config.quantum_depth, 8, len(config.rotation_gates)))
        rotations = {"RX": qml.RX, "RY": qml.RY, "RZ": qml.RZ}
        observable = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}[config.measurement]
        schedule = circuit_schedule(config, noise)

        def operations(inputs, weights):
            for gate, info, duration in schedule:
                if gate in ("CNOT", "CZ"):
                    {"CNOT": qml.CNOT, "CZ": qml.CZ}[gate](wires=info)
                elif gate != "idle":
                    for wire in range(8):
                        if gate == "H":
                            qml.Hadamard(wire)
                        elif gate == "embed":
                            rotations[config.embedding](torch.pi * inputs[..., wire], wires=wire)
                        else:
                            layer, gate_index = info
                            rotations[gate](weights[layer, wire, gate_index], wires=wire)
                if noise.enabled and duration:
                    for wire in range(8):
                        qml.ThermalRelaxationError(0.0, noise.t1_us[wire], noise.t2_us[wire], duration, wires=wire)
        strict_shots = bool(self.shots and config.measurement_mode == "parameter_shift")

        @qml.qnode(qml.device("default.mixed", wires=8, seed=self.sampling_rng), interface="torch",
                   diff_method="parameter-shift" if strict_shots else "backprop")
        def circuit(inputs, weights):
            operations(inputs, weights)
            return tuple(qml.expval(observable(wire)) for wire in range(8))

        self.circuit = qml.set_shots(circuit, shots=self.shots) if strict_shots else circuit

        @qml.qnode(qml.device("default.mixed", wires=8, seed=self.sampling_rng), interface="torch", diff_method="backprop")
        def probability_circuit(inputs, weights):
            operations(inputs, weights)
            rotate_measurement_basis(config.measurement)
            return qml.probs(wires=range(8))

        self.probability_circuit = probability_circuit

    def set_measurement_seed(self, seed):
        self.sampling_rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state
        self.backprop_sampling_rng.manual_seed(seed)

    def forward(self, inputs):
        if effective_shots(self.config, self.training) and self.config.measurement_mode != "parameter_shift":
            probabilities = self.probability_circuit(inputs, self.weights)
            return readout_with_analytic_jacobian(probabilities, self.shots, self.backprop_sampling_rng).to(inputs.dtype)
        if self.shots and self.config.measurement_mode == "parameter_shift":
            return torch.stack([torch.stack(self.circuit(row, self.weights)) for row in inputs]).to(inputs.dtype)
        return torch.stack(self.circuit(inputs, self.weights), dim=-1).to(inputs.dtype)


class NoisyHybridClassifier(HybridClassifier):
    def __init__(self, config: ModelConfig, noise: NoiseConfig):
        super().__init__(config)
        self.noise = noise
        if noise.backend == "pennylane_statevector":
            self.quantum = QuantumBlock(config.quantum_depth, config)
        else:
            block = TorchDensityBlock if noise.backend == "torch_density" else PennyLaneMixedBlock
            self.quantum = block(config, noise)
        self.to(dtype=torch.float64 if noise.precision == "float64" else torch.float32)
        self.set_mode()

    def noise_dict(self):
        return asdict(self.noise)
