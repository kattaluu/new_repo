"""Fixed coherent calibration errors, with exact average gate fidelity.

RX/RY/RZ use additive overrotation around their own axis; H uses a following Z
rotation; CNOT/CZ/iSWAP use a following ZZ rotation. Signs are fixed per gate/wire or
ordered edge. This is a calibration realization, not stochastic gate failures.
"""

import math
import numpy as np


def ideal_entangler(gate):
    if gate == "CNOT":
        return np.eye(4, dtype=complex)[[0, 1, 3, 2]]
    if gate == "CZ":
        return np.diag([1, 1, 1, -1]).astype(complex)
    if gate == "iSWAP":
        return np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex)
    raise ValueError(f"Unknown entangler: {gate}")


def fidelity_angle(fidelity, dimension):
    # F_avg(E U, U) = (|Tr(E)|**2 + d)/(d*(d+1)), E=exp(-i*angle*P/2).
    return 2 * math.asin(math.sqrt((1 - fidelity) * (dimension + 1) / dimension))


class CoherentGateErrors:
    def __init__(self, single_fidelity, two_fidelity, seed):
        rng = np.random.default_rng(seed)
        self.offsets = dict(zip(("RX", "RY", "RZ", "H"),
                               rng.choice([-1.0, 1.0], size=(4, 8)) * fidelity_angle(single_fidelity, 2)))
        self.two_angles = dict(zip(("CNOT", "CZ"),
                                  rng.choice([-1.0, 1.0], size=(2, 8, 8)) * fidelity_angle(two_fidelity, 4)))
        self.two_angles["iSWAP"] = rng.choice([-1.0, 1.0], size=(8, 8)) * fidelity_angle(two_fidelity, 4)
        h = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        self.hadamards = tuple(np.diag(np.exp(-0.5j * angle * np.array([1, -1]))) @ h
                               for angle in self.offsets["H"])
        self.entanglers = {}
        for gate in ("CNOT", "CZ", "iSWAP"):
            for control in range(8):
                for target in range(8):
                    if control == target:
                        continue
                    ideal = ideal_entangler(gate)
                    angle = self.two_angles[gate][control, target]
                    error = np.diag(np.exp(-0.5j * angle * np.array([1, -1, -1, 1])))
                    self.entanglers[gate, control, target] = error @ ideal

    def offset(self, gate, wire):
        return float(self.offsets[gate][wire])

    def entangler(self, gate, edge):
        return self.entanglers[gate, *edge]


def compile_gate_errors(noise):
    if not noise.gate_errors:
        return None
    return CoherentGateErrors(noise.single_gate_fidelity, noise.two_gate_fidelity, noise.gate_error_seed)
