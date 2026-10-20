"""CPU Aer circuits with a first-order PyTorch parameter-shift bridge.

Each rotation occurrence has its own parameter, including repeated encodings.
PyTorch then applies the chain rule to shared weights, inputs and embedding scale.
Finite shots sample joint bitstrings, exactly as in the other project backends.
"""

import math
import numpy as np
import torch
from torch import nn
from torch.autograd.function import once_differentiable

from .gate_errors import compile_gate_errors
from .measurements import effective_shots, pauli_signs, sample_expectations


class AerBlock(nn.Module):
    def __init__(self, config, noise):
        super().__init__()
        try:
            from qiskit import QuantumCircuit, transpile
            from qiskit.circuit import ParameterVector
            from qiskit_aer import AerSimulator
            from qiskit_aer.noise import thermal_relaxation_error
        except ImportError as exc:
            raise ImportError('Install Aer with: python -m pip install -e ".[aer]"') from exc
        from .noisy import circuit_schedule

        self.config, self.noise = config, noise
        self.weights = nn.Parameter(0.05 * torch.randn(config.quantum_depth, 8, len(config.rotation_gates)))
        self.register_parameter("embedding_scale", nn.Parameter(torch.ones(8)) if config.trainable_embedding else None)
        self.sampling_rng = torch.Generator().manual_seed(config.measurement_seed)
        method = "statevector" if noise.backend == "aer_statevector" else "density_matrix"
        # Outer CPUEngine owns parallelism: avoid workers * all CPU cores.
        self.simulator = AerSimulator(method=method, precision="single" if noise.precision == "float32" else "double",
                                      max_parallel_threads=1, max_parallel_experiments=1, max_parallel_shots=1)
        schedule = circuit_schedule(config, noise)
        errors = compile_gate_errors(noise)
        self.occurrences = [(gate, info, wire) for gate, info, _ in schedule
                            if gate in ("embed", "RX", "RY", "RZ") for wire in range(8)]
        self.angle_parameters = ParameterVector("angle", len(self.occurrences))
        circuit = QuantumCircuit(8)
        offset = 0
        thermal = {}
        for gate, info, duration in schedule:
            if gate.removesuffix("_parallel") in ("CNOT", "CZ"):
                kind = gate.removesuffix("_parallel")
                for control, target in (info if gate.endswith("_parallel") else (info,)):
                    # Logical wire zero is the most significant bit throughout this project.
                    getattr(circuit, "cx" if kind == "CNOT" else "cz")(7 - control, 7 - target)
                    if errors:
                        circuit.rzz(float(errors.two_angles[kind][control, target]), 7 - control, 7 - target)
            elif gate == "H":
                for wire in range(8):
                    circuit.h(7 - wire)
                    if errors:
                        circuit.rz(errors.offset("H", wire), 7 - wire)
            elif gate != "idle":
                kind = config.embedding if gate == "embed" else gate
                for wire in range(8):
                    theta = self.angle_parameters[offset] + (errors.offset(kind, wire) if errors else 0)
                    getattr(circuit, kind.lower())(theta, 7 - wire)
                    offset += 1
            if noise.enabled and duration:
                for wire in range(8):
                    key = (noise.t1_us[wire], noise.t2_us[wire], duration)
                    if key not in thermal:
                        thermal[key] = thermal_relaxation_error(*key, excited_state_population=0).to_instruction()
                    circuit.append(thermal[key], [7 - wire])
        for wire in range(8):
            if config.measurement == "Y":
                circuit.sdg(7 - wire)
            if config.measurement in ("X", "Y"):
                circuit.h(7 - wire)
        circuit.save_probabilities(label="probabilities")
        # Compile once; never recompile on every training example or shift.
        self.circuit = transpile(circuit, self.simulator, optimization_level=0)

    def set_measurement_seed(self, seed):
        self.sampling_rng.manual_seed(seed)

    def _evaluate(self, angles, shots):
        values = angles.detach().cpu().double().numpy()
        outputs = []
        # Bound memory for batches of shifted circuits and their density matrices.
        for start in range(0, len(values), 64):
            circuits = [self.circuit.assign_parameters(dict(zip(self.angle_parameters, row)), inplace=False)
                        for row in values[start:start + 64]]
            result = self.simulator.run(circuits, shots=1).result()
            if not result.success:
                raise RuntimeError(f"Aer simulation failed: {result.status}")
            probabilities = torch.from_numpy(np.stack([result.data(i)["probabilities"] for i in range(len(circuits))])).to(angles.dtype)
            if shots is None:
                outputs.append(probabilities @ pauli_signs().to(angles.dtype))
            else:
                outputs.append(sample_expectations(probabilities, shots, self.sampling_rng))
        return torch.cat(outputs).to(angles.device)

    def forward(self, inputs):
        if inputs.device.type != "cpu":
            raise ValueError("Aer backends currently support CPU tensors only")
        if self.embedding_scale is not None:
            inputs = inputs * self.embedding_scale
        columns = []
        for gate, info, wire in self.occurrences:
            if gate == "embed":
                columns.append(math.pi * inputs[:, wire])
            else:
                layer, gate_index = info
                columns.append(self.weights[layer, wire, gate_index].expand(len(inputs)))
        angles = torch.stack(columns, dim=1)
        shots = effective_shots(self.config, self.training)
        gradient_shots = shots if self.config.measurement_mode == "parameter_shift" else None
        return _AerExpectation.apply(angles, self, shots, gradient_shots)


class _AerExpectation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, angles, block, shots, gradient_shots):
        ctx.save_for_backward(angles)
        ctx.block, ctx.gradient_shots = block, gradient_shots
        return block._evaluate(angles, shots)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        (angles,) = ctx.saved_tensors
        gradient = torch.zeros_like(angles)
        # Shift individual gate occurrences, not shared input variables. This also
        # handles data re-uploading and repeated RX/RY/RZ layers correctly.
        for column in range(angles.shape[1]):
            plus, minus = angles.clone(), angles.clone()
            plus[:, column] += math.pi / 2
            minus[:, column] -= math.pi / 2
            values = ctx.block._evaluate(torch.cat((plus, minus)), ctx.gradient_shots)
            derivative = (values[:len(angles)] - values[len(angles):]) / 2
            gradient[:, column] = (derivative * upstream).sum(-1)
        return gradient, None, None, None
