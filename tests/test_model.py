import itertools

import numpy as np
import pennylane as qml
import pytest
import torch
from torch import nn

from qml_beans.model import HybridClassifier, ModelConfig, QuantumBlock


torch.set_num_threads(1)


@pytest.mark.parametrize("quantum,classical_train,quantum_train", list(itertools.product([False, True], repeat=3)))
def test_switches_update_only_authorized_parameters(quantum, classical_train, quantum_train):
    torch.manual_seed(7)
    model = HybridClassifier(ModelConfig(5, 3, quantum_depth=1))
    model.set_mode(quantum=quantum, train_classical=classical_train, train_quantum=quantum_train)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    features = torch.randn(6, 5)
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    logits = model(features)
    assert logits.shape == (6, 3)
    assert torch.isfinite(logits).all()
    params = [p for p in model.parameters() if p.requires_grad]
    if params:
        optimizer = torch.optim.SGD(params, lr=0.2)
        nn.functional.cross_entropy(logits, labels).backward()
        optimizer.step()
    changed = {name: not torch.equal(before[name], p) for name, p in model.named_parameters()}
    for prefix, enabled in (("encoder.", classical_train), ("head.", classical_train),
                            ("quantum.", quantum and quantum_train)):
        changes = [value for name, value in changed.items() if name.startswith(prefix)]
        assert any(changes) == enabled
    if quantum and classical_train and not quantum_train:
        # Freezing the circuit must not sever encoder gradients.
        assert model.encoder[0].weight.grad.abs().sum() > 0


def test_state_vector_matches_independent_pennylane_reference():
    torch.manual_seed(9)
    block = QuantumBlock(2).double()
    inputs = torch.randn(3, 8, dtype=torch.float64) * 0.2
    weights = block.weights.detach().numpy()

    @qml.qnode(qml.device("default.qubit", wires=8))
    def reference(x):
        for wire in range(8):
            qml.RY(np.pi * x[wire], wires=wire)
        for layer in weights:
            for wire in range(8):
                qml.RY(layer[wire, 0], wires=wire)
                qml.RX(layer[wire, 1], wires=wire)
            for wire in range(8):
                qml.CNOT(wires=[wire, (wire + 1) % 8])
        return qml.state()

    actual = block(inputs).detach().numpy()
    basis = np.arange(256)
    for i, x in enumerate(inputs.numpy()):
        state = reference(x)
        assert np.isclose(np.abs(state).dot(np.abs(state)), 1)
        expected = [np.sum(np.abs(state) ** 2 * (1 - 2 * ((basis >> (7 - wire)) & 1)))
                    for wire in range(8)]
        np.testing.assert_allclose(actual[i], expected, atol=1e-10)


def test_quantum_gradients_match_finite_differences():
    torch.manual_seed(3)
    block = QuantumBlock(1).double()
    inputs = (torch.randn(2, 8, dtype=torch.float64) * 0.1).requires_grad_()
    block(inputs).sum().backward()
    analytic_input = inputs.grad[0, 2].item()
    analytic_weight = block.weights.grad[0, 3, 0].item()
    epsilon = 1e-6
    with torch.no_grad():
        plus, minus = inputs.clone(), inputs.clone()
        plus[0, 2] += epsilon
        minus[0, 2] -= epsilon
        numeric_input = ((block(plus).sum() - block(minus).sum()) / (2 * epsilon)).item()
        original = block.weights[0, 3, 0].item()
        block.weights[0, 3, 0] = original + epsilon
        positive = block(inputs).sum().item()
        block.weights[0, 3, 0] = original - epsilon
        negative = block(inputs).sum().item()
        block.weights[0, 3, 0] = original
    assert analytic_input == pytest.approx(numeric_input, abs=1e-6)
    assert analytic_weight == pytest.approx((positive - negative) / (2 * epsilon), abs=1e-6)


def test_bypass_never_executes_circuit(monkeypatch):
    model = HybridClassifier(ModelConfig(4, 2))
    model.set_mode(quantum=False)

    def fail(*args):
        raise AssertionError("Quantum circuit should not be executed")

    monkeypatch.setattr(model.quantum, "forward", fail)
    assert model(torch.randn(1, 4)).shape == (1, 2)
