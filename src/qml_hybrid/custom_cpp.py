"""Direct CustomDevice kernels with an analytic reverse-mode PyTorch bridge.

Does not import the obsolete PennyLane 0.25 wrappers. Native kernels use complex128.
Unitary forward/reverse evolution runs in C++; thermal adjoints use NumPy block
formulas. No shifted circuits are needed for analytic/surrogate/REINFORCE backward.
"""

import importlib.machinery
import importlib.util
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

from .gate_errors import compile_gate_errors, ideal_entangler
from .encoding import encode_features, encoding_index, encoding_parameter
from .measurements import (effective_shots, pauli_signs, readout_with_analytic_jacobian,
                           reinforce_readout, sample_expectations)


def custom_device_directory(value=""):
    candidate = value or os.environ.get("QML_CUSTOM_DEVICE_PATH")
    if not candidate:
        candidate = Path(__file__).resolve().parents[2] / "native/custom_device"
    root = Path(candidate).expanduser().resolve()
    return root / "cpp builds" if (root / "cpp builds").is_dir() else root


def load_custom_engine(kind, directory=""):
    root = custom_device_directory(directory)
    name = "engine8q_" + kind
    candidates = [root / (name + suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES]
    binary = next((path for path in candidates if path.is_file()), None)
    if binary is None:
        raise ImportError(f"Не найден C++-модуль {name} для Python {sys.version_info.major}.{sys.version_info.minor}: "
                          f"{root}. Укажите CustomDevice или cpp builds; нужна сборка для текущего Python/ОС. "
                          "Старые .so/.pyd другой версии Python не подходят.")
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != binary.resolve():
            raise ImportError(f"{name} уже загружен из другой папки. Перезапустите ядро для смены сборки.")
        return module.Engine
    try:
        spec = importlib.util.spec_from_file_location(name, binary)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except (ImportError, OSError) as exc:
        raise ImportError(f"Не удалось загрузить {binary}: {exc}") from exc
    sys.modules[name] = module
    return module.Engine


def _pauli_left(state, gate, wire):
    indices = np.arange(256)
    bits = (indices >> (7 - wire)) & 1
    if gate == "RZ":
        result = state
        factors = 1 - 2 * bits
    else:
        result = state[indices ^ (1 << (7 - wire))]
        factors = np.ones(256) if gate == "RX" else -1j * (1 - 2 * bits)
    return result * (factors[:, None] if state.ndim == 2 else factors)


def _thermal_adjoint(matrix, coefficients):
    # E*(A)00=A00; E*(A)11=(1-e1)A00+e1*A11; off-diagonal=e2*A01.
    for wire, (e1, e2) in enumerate(coefficients):
        blocks = np.moveaxis(matrix.reshape([2] * 16), (wire, wire + 8), (0, 1))
        result = blocks.copy()
        result[1, 1] = (1 - e1) * blocks[0, 0] + e1 * blocks[1, 1]
        result[0, 1] = e2 * blocks[0, 1]
        result[1, 0] = e2 * blocks[1, 0]
        matrix = np.moveaxis(result, (0, 1), (wire, wire + 8)).reshape(256, 256)
    return np.ascontiguousarray(matrix)


class _CppProbabilities(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, block):
        ctx.block = block
        ctx.save_for_backward(inputs, weights)
        need = tuple(ctx.needs_input_grad[:2])
        probabilities, ctx.tapes = block._probabilities_numpy(inputs, weights, need=need)
        return torch.from_numpy(probabilities).to(dtype=inputs.dtype)

    @staticmethod
    def backward(ctx, upstream):
        inputs, weights = ctx.saved_tensors
        need_input, need_weights = ctx.needs_input_grad[:2]
        gx = np.zeros(tuple(inputs.shape)) if need_input else None
        gw = np.zeros(tuple(weights.shape)) if need_weights else None
        upstream = upstream.detach().double().numpy()
        block = ctx.block
        for sample, (tape, final_state) in enumerate(ctx.tapes):
            initial = (np.diag(upstream[sample]).astype(complex) if block.kind == "dm"
                       else upstream[sample] * final_state)
            adjoint = block._engine_from_state(initial)
            for step in reversed(tape):
                if step[0] == "thermal":
                    state = _thermal_adjoint(np.asarray(adjoint.get_state()), step[1])
                    adjoint = block._engine_from_state(state)
                elif step[0] == "unitary":
                    _, matrix, wires, fast = step
                    if fast == "CNOT":
                        adjoint.apply_cnot(*wires)
                    elif fast == "CZ":
                        adjoint.apply_cphase(-math.pi, *wires)
                    else:
                        adjoint.apply_unitary(np.ascontiguousarray(matrix.conj().T), wires)
                else:
                    _, gate, angle, wire, parameter, state = step
                    if state is not None:
                        value = np.vdot(np.asarray(adjoint.get_state()), _pauli_left(state, gate, wire)).imag
                        if parameter[0] == "input":
                            gx[sample, parameter[1]] += math.pi * value
                        else:
                            _, layer, gate_index = parameter
                            gw[layer, wire, gate_index] += value
                    getattr(adjoint, "apply_" + gate.lower())(-angle, wire)
        return (torch.from_numpy(gx).to(inputs.dtype) if gx is not None else None,
                torch.from_numpy(gw).to(weights.dtype) if gw is not None else None, None)


class CustomCppBlock(nn.Module):
    def __init__(self, config, noise):
        super().__init__()
        from .noisy import circuit_schedule
        self.config, self.noise = config, noise
        self.kind = "dm" if noise.backend == "custom_cpp_dm" else "sv"
        self.engine_class = load_custom_engine(self.kind, noise.custom_device_path)
        self._rng_state = self.engine_class().__getstate__()[1]
        self.weights = nn.Parameter(0.05 * torch.randn(config.quantum_depth, 8, len(config.rotation_gates)))
        self.register_parameter("embedding_scale", encoding_parameter(config))
        self.register_buffer("z_signs", pauli_signs().float())
        self.schedule = circuit_schedule(config, noise)
        self.gate_errors = compile_gate_errors(noise)
        self.sampling_rng = torch.Generator().manual_seed(config.measurement_seed)
        self._hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        self._two = {tuple(edge): (self.gate_errors.entangler(config.entangler, edge) if self.gate_errors
                                 else ideal_entangler(config.entangler)) for edge in config.edges()}
        self._thermal = {}
        if noise.enabled:
            for _, _, duration in self.schedule:
                if duration:
                    self._thermal[duration] = tuple((math.exp(-duration / t1), math.exp(-duration / t2))
                                                    for t1, t2 in zip(noise.t1_us, noise.t2_us))

    def set_measurement_seed(self, seed):
        self.sampling_rng.manual_seed(seed)

    def _engine_from_state(self, state):
        engine = self.engine_class.__new__(self.engine_class)
        engine.__setstate__((np.ascontiguousarray(state, dtype=np.complex128), self._rng_state))
        return engine

    def _run_one(self, inputs, weights, *, need=(False, False), shift=None):
        engine = self.engine_class()
        tape = [] if any(need) else None

        def fixed(matrix, wires, fast=None):
            if fast == "CNOT":
                engine.apply_cnot(*wires)
            elif fast == "CZ":
                engine.apply_cphase(math.pi, *wires)
            else:
                engine.apply_unitary(matrix, list(wires))
            if tape is not None:
                tape.append(("unitary", matrix, list(wires), fast))

        occurrence = -1
        for moment, (gate, info, duration) in enumerate(self.schedule):
            if gate == "embed":
                occurrence += 1
            kind = gate.removesuffix("_parallel")
            if kind in ("CNOT", "CZ", "iSWAP"):
                for edge in info if gate.endswith("_parallel") else (info,):
                    fixed(self._two[tuple(edge)], edge,
                          kind if not self.gate_errors and kind in ("CNOT", "CZ") else None)
            elif gate == "H":
                for wire in range(8):
                    fixed(self.gate_errors.hadamards[wire] if self.gate_errors else self._hadamard, (wire,))
            elif gate != "idle":
                for wire in range(8):
                    if gate == "embed":
                        axis = self.config.embedding
                        index = encoding_index(self.config, occurrence, wire)
                        angle, parameter = math.pi * inputs[index], ("input", index)
                        retain = need[0]
                    else:
                        axis = gate
                        layer, gate_index = info
                        angle, parameter = weights[layer, wire, gate_index], ("weight", layer, gate_index)
                        retain = need[1]
                    if self.gate_errors:
                        angle += self.gate_errors.offset(axis, wire)
                    if shift is not None and shift[:2] == (moment, wire):
                        angle += shift[2]
                    getattr(engine, "apply_" + axis.lower())(float(angle), wire)
                    if tape is not None:
                        tape.append(("rotation", axis, float(angle), wire, parameter,
                                     np.asarray(engine.get_state()) if retain else None))
            if self.noise.enabled and duration:
                coefficients = self._thermal[duration]
                for wire, ((e1, _), t1, t2) in enumerate(zip(coefficients, self.noise.t1_us, self.noise.t2_us)):
                    engine.apply_amplitude_damping(wire, 1 - e1)
                    # T2 already contains T1: add only pure dephasing.
                    gamma = -math.expm1(-2 * duration * max(0.0, 1 / t2 - 1 / (2 * t1)))
                    if gamma:
                        engine.apply_phase_damping(wire, gamma)
                if tape is not None:
                    tape.append(("thermal", coefficients))
        if self.config.measurement in ("X", "Y"):
            basis = (self._hadamard if self.config.measurement == "X"
                     else self._hadamard @ np.diag([1, -1j]))
            for wire in range(8):
                fixed(basis, (wire,))
        probabilities = np.asarray(engine.get_probs())
        final_state = np.asarray(engine.get_state()) if tape is not None and self.kind == "sv" else None
        return probabilities, (tape, final_state)

    def _probabilities_numpy(self, inputs, weights, *, need=(False, False), shift=None):
        if inputs.device.type != "cpu":
            raise ValueError("CustomDevice C++ поддерживает только CPU")
        x, w = inputs.detach().double().numpy(), weights.detach().double().numpy()
        results = [self._run_one(row, w, need=need, shift=shift) for row in x]
        return np.stack([result[0] for result in results]), [result[1] for result in results]

    def _measure(self, inputs, weights=None, shift=None):
        weights = self.weights if weights is None else weights
        if shift is None and torch.is_grad_enabled() and (inputs.requires_grad or weights.requires_grad):
            probabilities = _CppProbabilities.apply(inputs, weights, self)
        else:
            probabilities = torch.from_numpy(self._probabilities_numpy(inputs, weights, shift=shift)[0]).to(inputs.dtype)
        shots = effective_shots(self.config, self.training)
        if not shots:
            return probabilities @ self.z_signs.to(inputs.dtype)
        if self.config.measurement_mode == "reinforce":
            return reinforce_readout(self, probabilities, shots, self.sampling_rng)
        if self.config.measurement_mode == "parameter_shift":
            return sample_expectations(probabilities, shots, self.sampling_rng)
        return readout_with_analytic_jacobian(probabilities, shots, self.sampling_rng)

    def forward(self, inputs):
        from .noisy import _ShotParameterShift
        self._reinforce_score = self._reinforce_mean = None
        inputs = encode_features(inputs, self.embedding_scale, self.config)
        if self.config.shots and self.config.measurement_mode == "parameter_shift":
            return _ShotParameterShift.apply(inputs, self.weights, self)
        return self._measure(inputs)
