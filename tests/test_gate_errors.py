from dataclasses import asdict, replace

import numpy as np
import pytest
import torch
import pennylane as qml

from qml_beans.gate_errors import CoherentGateErrors
from qml_beans.model import ModelConfig, QuantumBlock
from qml_beans.noisy import NoiseConfig, NoisyHybridClassifier, TorchDensityBlock, PennyLaneMixedBlock
from qml_beans.vision_training import CPUConfig, CPUEngine, load_noisy_checkpoint


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def average_fidelity(actual, ideal):
    d = len(ideal)
    return (abs(np.trace(ideal.conj().T @ actual)) ** 2 + d) / (d * (d + 1))


def test_compiled_gates_fidelity_and_local_rng():
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    errors = CoherentGateErrors(.99, .90, 42)
    assert all(np.array_equal(a, b) for a, b in zip(np_state, np.random.get_state()))
    assert torch.equal(torch_state, torch.get_rng_state())
    for gate in ('RX', 'RY', 'RZ'):
        ideal = qml.matrix(getattr(qml, gate)(.73, wires=0))
        for wire in range(8):
            actual = qml.matrix(getattr(qml, gate)(.73 + errors.offset(gate, wire), wires=0))
            assert average_fidelity(actual, ideal) == pytest.approx(.99, abs=1e-14)
    for wire in range(8):
        assert average_fidelity(errors.hadamards[wire], qml.matrix(qml.Hadamard(0))) == pytest.approx(.99)
    for (gate, control, target), actual in errors.entanglers.items():
        ideal = qml.matrix(getattr(qml, gate)(wires=[0, 1]))
        assert average_fidelity(actual, ideal) == pytest.approx(.90, abs=1e-14)
        np.testing.assert_allclose(actual.conj().T @ actual, np.eye(4), atol=1e-14)
    other = CoherentGateErrors(.99, .90, 42)
    for gate in errors.offsets:
        np.testing.assert_array_equal(errors.offsets[gate], other.offsets[gate])
    assert any(not np.array_equal(errors.offsets[gate], CoherentGateErrors(.99, .90, 43).offsets[gate])
               for gate in errors.offsets)


@pytest.mark.parametrize('topology,entangler,measurement', [
    ('linear', 'CNOT', 'Z'), ('brickwork', 'CNOT', 'X'), ('brickwork', 'CZ', 'Y'),
])
def test_compiled_density_matches_pennylane_and_statevector(topology, entangler, measurement):
    config = ModelConfig(8, 2, quantum_depth=2, reupload=True, initial_hadamard=True,
                         entanglement=topology, entangler=entangler, measurement=measurement)
    noise = NoiseConfig(gate_errors=True, precision='float64')
    a, b = TorchDensityBlock(config, noise).double(), PennyLaneMixedBlock(config, noise).double()
    b.weights.data.copy_(a.weights)
    x = torch.randn(1, 8, dtype=torch.float64, requires_grad=True)
    ya, yb = a(x), b(x)
    torch.testing.assert_close(ya, yb, atol=2e-8, rtol=2e-8)
    ga = torch.autograd.grad(ya.square().sum(), (x, a.weights))
    gb = torch.autograd.grad(yb.square().sum(), (x, b.weights))
    for left, right in zip(ga, gb):
        torch.testing.assert_close(left, right, atol=2e-7, rtol=2e-7)
    ideal_noise = replace(noise, enabled=False)
    density = NoisyHybridClassifier(config, ideal_noise)
    statevector = NoisyHybridClassifier(config, replace(ideal_noise, backend='pennylane_statevector'))
    statevector.load_state_dict({k: v for k, v in density.state_dict().items() if k != 'quantum.z_signs'})
    torch.testing.assert_close(density(x), statevector(x), atol=1e-10, rtol=1e-10)
    rho = density.quantum.density_matrix(x).detach()
    torch.testing.assert_close(rho, rho.mH, atol=1e-12, rtol=1e-12)
    assert rho[0].trace().real == pytest.approx(1)


@pytest.mark.parametrize('backend', ['torch_density', 'pennylane_mixed', 'pennylane_statevector'])
def test_unit_fidelity_and_checkpoint_roundtrip(backend, tmp_path):
    config = ModelConfig(8, 2, quantum_depth=1, entanglement='brickwork', initial_hadamard=True)
    noise = NoiseConfig(backend=backend, enabled=False, precision='float64')
    clean = NoisyHybridClassifier(config, noise)
    errors = NoisyHybridClassifier(config, replace(noise, gate_errors=True, single_gate_fidelity=1., two_gate_fidelity=1.))
    errors.load_state_dict(clean.state_dict())
    x = torch.randn(1, 8, dtype=torch.float64)
    torch.testing.assert_close(errors(x), clean(x), atol=1e-12, rtol=1e-12)
    actual = NoisyHybridClassifier(config, replace(noise, gate_errors=True))
    path = tmp_path/'best.pt'
    torch.save(dict(format='noisy_vision_v1', model_config=actual.config_dict(), noise_config=actual.noise_dict(),
                    state_dict=actual.state_dict(), mode=actual.mode_dict()), path)
    restored, _ = load_noisy_checkpoint(path)
    torch.testing.assert_close(actual(x), restored(x), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('backend', ['torch_density', 'pennylane_mixed', 'pennylane_statevector'])
def test_gate_errors_finite_shot_parameter_shift(backend):
    config = ModelConfig(8, 2, quantum_depth=1, rotation_gates=('RX',), shots=32,
                         entanglement='custom', custom_edges=((0, 1),), initial_hadamard=True)
    model = NoisyHybridClassifier(config, NoiseConfig(backend=backend, enabled=False, gate_errors=True))
    model(torch.rand(1, 8)).square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_compiled_errors_parallel_cpu():
    config = ModelConfig(8, 2, quantum_depth=1, entanglement='brickwork')
    noise = NoiseConfig(gate_errors=True, precision='float64')
    a, b = NoisyHybridClassifier(config, noise), NoisyHybridClassifier(config, noise)
    b.load_state_dict(a.state_dict())
    data = {'train': (torch.randn(3, 8), torch.tensor([0, 1, 0]))}
    with CPUEngine(a, data, CPUConfig(microbatch_size=1)) as engine:
        loss_a, logits_a = engine.run('train', [0, 1, 2], backward=True)
    with CPUEngine(b, data, CPUConfig(workers=2, microbatch_size=1)) as engine:
        loss_b, logits_b = engine.run('train', [0, 1, 2], backward=True)
    assert loss_a == pytest.approx(loss_b, abs=1e-10)
    torch.testing.assert_close(logits_a, logits_b, atol=1e-10, rtol=1e-10)
    for left, right in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('kwargs', [dict(single_gate_fidelity=1.1), dict(two_gate_fidelity=float('nan')),
                                   dict(gate_error_seed=-1), dict(gate_errors='yes')])
def test_invalid_gate_error_config(kwargs):
    with pytest.raises(ValueError):
        NoiseConfig(**kwargs)
