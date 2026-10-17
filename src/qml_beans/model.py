"""An eight-dimensional interface allows switching the circuit on and off."""

from dataclasses import asdict, dataclass

import pennylane as qml
import numpy as np
import torch
from torch import nn

from .measurements import (MEASUREMENT_MODES, effective_shots, readout_with_analytic_jacobian,
                           rotate_measurement_basis)


ACTIVATIONS = {
    "relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid,
    "gelu": nn.GELU, "silu": nn.SiLU, "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU, "identity": nn.Identity,
}


@dataclass(frozen=True)
class ModelConfig:
    n_features: int
    n_classes: int
    pre_hidden: int = 32
    post_hidden: int = 16
    quantum_depth: int = 2
    n_qubits: int = 8
    # None retains the original architecture and old checkpoint parameter names.
    pre_layers: tuple[int, ...] | None = None
    post_layers: tuple[int, ...] | None = None
    pre_activations: tuple[str, ...] | None = None
    post_activations: tuple[str, ...] | None = None
    latent_activation: str = "tanh"
    embedding: str = "RY"
    rotation_gates: tuple[str, ...] = ("RY", "RX")
    entanglement: str = "ring"
    entangler: str = "CNOT"
    custom_edges: tuple[tuple[int, int], ...] = ()
    reupload: bool = False
    initial_hadamard: bool = False
    measurement: str = "Z"
    shots: int | None = None
    measurement_seed: int = 42
    measurement_mode: str = "parameter_shift"

    def __post_init__(self):
        if self.measurement_mode not in MEASUREMENT_MODES:
            raise ValueError(f"measurement_mode must be one of {MEASUREMENT_MODES}")
        if self.shots is not None and (type(self.shots) is not int or self.shots < 1):
            raise ValueError("shots must be None (analytic) or a positive integer")
        if type(self.measurement_seed) is not int or not 0 <= self.measurement_seed < 2**32:
            raise ValueError("measurement_seed must be an integer in 0..2**32-1")
        if self.n_qubits != 8:
            raise ValueError("This architecture requires exactly 8 qubits.")
        if min(self.n_features, self.pre_hidden, self.post_hidden, self.quantum_depth) < 1:
            raise ValueError("Feature count, hidden sizes and circuit depth must be positive.")
        if self.n_classes < 2:
            raise ValueError("Classification requires at least two classes.")
        for name in ("pre_layers", "post_layers", "pre_activations", "post_activations", "rotation_gates"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(value))
        object.__setattr__(self, "custom_edges", tuple(tuple(edge) for edge in self.custom_edges))
        for widths, activations in ((self.encoder_widths, self.encoder_activations),
                                     (self.head_widths, self.head_activations)):
            if any(type(width) is not int or width < 1 for width in widths):
                raise ValueError("Hidden layer widths must be positive integers")
            if len(widths) != len(activations):
                raise ValueError("Specify one activation for each hidden layer")
            if any(activation not in ACTIVATIONS for activation in activations):
                raise ValueError(f"Supported activations: {tuple(ACTIVATIONS)}")
        if self.latent_activation not in ACTIVATIONS:
            raise ValueError("Unsupported latent activation")
        if self.embedding not in ("RX", "RY", "RZ"):
            raise ValueError("Embedding must be RX, RY or RZ")
        if self.embedding == "RZ" and not self.initial_hadamard:
            raise ValueError("RZ encoding needs initial Hadamard gates to encode data in relative phases")
        if not self.rotation_gates or any(gate not in ("RX", "RY", "RZ") for gate in self.rotation_gates):
            raise ValueError("Choose a nonempty ordered sequence of RX / RY / RZ rotations")
        if self.entanglement not in ("ring", "linear", "brickwork", "full", "none", "custom"):
            raise ValueError(f"Unsupported entanglement topology: {self.entanglement!r}. "
                             "Choose ring, linear, brickwork, full, none or custom")
        if self.entangler not in ("CNOT", "CZ") or self.measurement not in ("X", "Y", "Z"):
            raise ValueError("Entangler must be CNOT/CZ and measurement must be X/Y/Z")
        for edge in self.custom_edges:
            if len(edge) != 2 or any(type(wire) is not int or not 0 <= wire < 8 for wire in edge):
                raise ValueError("Each custom edge needs two integer wire indices in 0..7")
            if edge[0] == edge[1]:
                raise ValueError("An entangling gate needs two different wires")

    @property
    def encoder_widths(self):
        return self.pre_layers if self.pre_layers is not None else (self.pre_hidden,)

    @property
    def head_widths(self):
        return self.post_layers if self.post_layers is not None else (self.post_hidden,)

    @property
    def encoder_activations(self):
        return self.pre_activations if self.pre_activations is not None else ("relu",) * len(self.encoder_widths)

    @property
    def head_activations(self):
        return self.post_activations if self.post_activations is not None else ("relu",) * len(self.head_widths)

    def edges(self):
        if self.entanglement == "brickwork":
            return ((0, 1), (2, 3), (4, 5), (6, 7), (1, 2), (3, 4), (5, 6))
        if self.entanglement == "ring":
            return tuple((wire, (wire + 1) % 8) for wire in range(8))
        if self.entanglement == "linear":
            return tuple((wire, wire + 1) for wire in range(7))
        if self.entanglement == "full":
            return tuple((left, right) for left in range(8) for right in range(left + 1, 8))
        return self.custom_edges if self.entanglement == "custom" else ()


def dense_block(n_inputs, widths, activations, n_outputs, output_activation=None):
    modules = []
    for width, activation in zip(widths, activations):
        modules.extend([nn.Linear(n_inputs, width), ACTIVATIONS[activation]()])
        n_inputs = width
    modules.append(nn.Linear(n_inputs, n_outputs))
    if output_activation is not None:
        modules.append(ACTIVATIONS[output_activation]())
    return nn.Sequential(*modules)


class QuantumBlock(nn.Module):
    """State vector with analytic, parameter-shift or sampled-forward/backprop readout."""

    def __init__(self, depth: int, config: ModelConfig | None = None, *, gate_errors=None):
        super().__init__()
        config = config or ModelConfig(8, 2, quantum_depth=depth)
        self.config = config
        self.shots = config.shots
        self.sampling_rng = np.random.default_rng(config.measurement_seed)
        self.backprop_sampling_rng = torch.Generator().manual_seed(config.measurement_seed)
        self.weights = nn.Parameter(0.05 * torch.randn(depth, 8, len(config.rotation_gates)))
        device = qml.device("default.qubit", wires=8, seed=self.sampling_rng)
        rotation_ops = {"RX": qml.RX, "RY": qml.RY, "RZ": qml.RZ}
        observable = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}[config.measurement]
        entangler = {"CNOT": qml.CNOT, "CZ": qml.CZ}[config.entangler]

        def embed(inputs):
            for wire in range(8):
                theta = torch.pi * inputs[..., wire]
                if gate_errors:
                    theta = theta + gate_errors.offset(config.embedding, wire)
                rotation_ops[config.embedding](theta, wires=wire)

        def operations(inputs, weights):
            if config.initial_hadamard:
                for wire in range(8):
                    if gate_errors:
                        qml.QubitUnitary(gate_errors.hadamards[wire], wires=wire)
                    else:
                        qml.Hadamard(wires=wire)
            embed(inputs)
            for layer in range(depth):
                if layer > 0 and config.reupload:
                    embed(inputs)
                for wire in range(8):
                    for gate_index, gate in enumerate(config.rotation_gates):
                        theta = weights[layer, wire, gate_index]
                        if gate_errors:
                            theta = theta + gate_errors.offset(gate, wire)
                        rotation_ops[gate](theta, wires=wire)
                for edge in config.edges():
                    if gate_errors:
                        qml.QubitUnitary(gate_errors.entangler(config.entangler, edge), wires=edge)
                    else:
                        entangler(wires=edge)

        strict_shots = bool(self.shots and config.measurement_mode == "parameter_shift")

        @qml.qnode(device, interface="torch", diff_method="parameter-shift" if strict_shots else "backprop")
        def circuit(inputs, weights):
            operations(inputs, weights)
            return tuple(qml.expval(observable(wire)) for wire in range(8))

        self.circuit = qml.set_shots(circuit, shots=self.shots) if strict_shots else circuit

        @qml.qnode(qml.device("default.qubit", wires=8, seed=self.sampling_rng), interface="torch", diff_method="backprop")
        def probability_circuit(inputs, weights):
            operations(inputs, weights)
            rotate_measurement_basis(config.measurement)
            return qml.probs(wires=range(8))

        self.probability_circuit = probability_circuit

    def set_measurement_seed(self, seed):
        self.sampling_rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state
        self.backprop_sampling_rng.manual_seed(seed)

    def forward(self, inputs):
        # PennyLane uses double precision internally; keep the classical model's dtype.
        if effective_shots(self.config, self.training) and self.config.measurement_mode != "parameter_shift":
            probabilities = self.probability_circuit(inputs, self.weights)
            return readout_with_analytic_jacobian(probabilities, self.shots, self.backprop_sampling_rng).to(inputs.dtype)
        if self.shots and self.config.measurement_mode == "parameter_shift":
            # Finite-shot parameter-shift must see individual tapes, not broadcasted inputs.
            return torch.stack([torch.stack(self.circuit(row, self.weights)) for row in inputs]).to(inputs.dtype)
        return torch.stack(self.circuit(inputs, self.weights), dim=-1).to(inputs.dtype)


class HybridClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = dense_block(config.n_features, config.encoder_widths,
                                   config.encoder_activations, 8, config.latent_activation)
        # Always instantiate it: checkpoints contain quantum weights even in bypass mode.
        self.quantum = QuantumBlock(config.quantum_depth, config)
        self.head = dense_block(8, config.head_widths, config.head_activations, config.n_classes)
        self.set_mode()

    def set_mode(self, *, quantum=True, train_classical=True, train_quantum=True):
        self.use_quantum = bool(quantum)
        self.train_classical = bool(train_classical)
        self.train_quantum = bool(train_quantum)
        for module in (self.encoder, self.head):
            module.requires_grad_(self.train_classical)
        self.quantum.requires_grad_(self.train_quantum and self.use_quantum)
        # Clear stale gradients when changing modes on an existing instance.
        for parameter in self.parameters():
            parameter.grad = None

    def mode_dict(self):
        return dict(quantum=self.use_quantum, train_classical=self.train_classical,
                    train_quantum=self.train_quantum)

    def config_dict(self):
        return asdict(self.config)

    def forward(self, features):
        latent = self.encoder(features)
        if self.use_quantum:
            # Do not detach: a frozen circuit must still pass gradients to the encoder.
            latent = self.quantum(latent)
        return self.head(latent)
