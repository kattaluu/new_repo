from dataclasses import replace

import pytest
import torch

from qml_beans.model import ModelConfig
from qml_beans.noisy import NoisyHybridClassifier, NoiseConfig
from qml_beans.vision_training import CPUConfig, CPUEngine, load_noisy_checkpoint


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def models(backend, mode, measurement="Z"):
    torch.manual_seed(17)
    config = ModelConfig(6, 3, shots=1000, measurement_mode=mode, quantum_depth=2, reupload=True,
                         rotation_gates=("RY", "RX"), measurement=measurement,
                         entanglement="custom", custom_edges=((0, 7), (1, 2)))
    noise = NoiseConfig(backend=backend, enabled=backend != "pennylane_statevector", precision="float64")
    actual = NoisyHybridClassifier(config, noise)
    analytic = NoisyHybridClassifier(replace(config, shots=None), noise)
    analytic.load_state_dict(actual.state_dict())
    return actual, analytic


@pytest.mark.parametrize("backend", ["torch_density", "pennylane_mixed", "pennylane_statevector"])
@pytest.mark.parametrize("measurement", ["X", "Y", "Z"])
def test_sampled_forward_analytic_vjp_all_backends_and_bases(backend, measurement):
    actual, analytic = models(backend, "sampled_backprop", measurement)
    x = torch.rand(2, 8, dtype=torch.float64, requires_grad=True)
    upstream = torch.linspace(-0.7, 1.2, 16, dtype=torch.float64).reshape(2, 8)
    sampled = actual.quantum(x)
    exact = analytic.quantum(x)
    counts = (sampled + 1) * 500
    torch.testing.assert_close(counts, counts.round(), atol=1e-10, rtol=0)
    assert (sampled - exact).abs().max() > 1e-5
    ga = torch.autograd.grad(sampled, (x, actual.quantum.weights), grad_outputs=upstream)
    gb = torch.autograd.grad(exact, (x, analytic.quantum.weights), grad_outputs=upstream)
    for left, right in zip(ga, gb):
        torch.testing.assert_close(left, right, atol=2e-9, rtol=2e-9)


@pytest.mark.parametrize("backend", ["torch_density", "pennylane_mixed", "pennylane_statevector"])
def test_analytic_train_mode_respects_train_eval_not_no_grad(backend, monkeypatch):
    actual, analytic = models(backend, "analytic_train")
    x = torch.rand(2, 8, dtype=torch.float64, requires_grad=True)
    # No sampling whatsoever during training, even in a no_grad/frozen training phase.
    with monkeypatch.context() as patch:
        def forbidden(*args, **kwargs):
            raise AssertionError("Analytic training must not sample")
        patch.setattr("qml_beans.measurements.sample_expectations", forbidden)
        patch.setattr("qml_beans.noisy.sample_expectations", forbidden)
        a, b = actual.quantum(x), analytic.quantum(x)
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
        ga = torch.autograd.grad(a.sum(), (x, actual.quantum.weights))
        gb = torch.autograd.grad(b.sum(), (x, analytic.quantum.weights))
        for left, right in zip(ga, gb):
            torch.testing.assert_close(left, right, atol=1e-11, rtol=1e-11)
        with torch.no_grad():
            torch.testing.assert_close(actual.quantum(x), b, atol=1e-12, rtol=1e-12)
    actual.eval()
    with torch.no_grad():
        actual.quantum.set_measurement_seed(11)
        first = actual.quantum(x)
        actual.quantum.set_measurement_seed(11)
        second = actual.quantum(x)
    assert torch.equal(first, second)
    counts = (first + 1) * 500
    torch.testing.assert_close(counts, counts.round(), atol=1e-10, rtol=0)
    assert (first - b).abs().max() > 1e-5
    actual.train()
    torch.testing.assert_close(actual.quantum(x), b, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("mode", ["sampled_backprop", "analytic_train"])
def test_end_to_end_encoder_through_frozen_quantum(mode):
    actual, _ = models("torch_density", mode)
    actual.set_mode(train_quantum=False)
    x = torch.randn(2, 6, dtype=torch.float64)
    loss = torch.nn.functional.cross_entropy(actual(x), torch.tensor([0, 2]))
    loss.backward()
    assert actual.quantum.weights.grad is None
    assert actual.encoder[0].weight.grad.abs().sum() > 0
    assert actual.head[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["sampled_backprop", "analytic_train"])
def test_measurement_mode_checkpoint_roundtrip(mode, tmp_path):
    actual, _ = models("torch_density", mode)
    path = tmp_path / "mode.pt"
    torch.save(dict(format="noisy_vision_v1", model_config=actual.config_dict(), noise_config=actual.noise_dict(),
                    mode=actual.mode_dict(), state_dict=actual.state_dict()), path)
    restored, _ = load_noisy_checkpoint(path)
    assert restored.config.measurement_mode == mode
    assert restored.config.shots == 1000
    actual.quantum.set_measurement_seed(77)
    restored.quantum.set_measurement_seed(77)
    x = torch.rand(2, 6, dtype=torch.float64)
    torch.testing.assert_close(actual(x), restored(x), atol=0, rtol=0)


def test_parallel_analytic_training_when_all_parameters_frozen():
    actual, _ = models("torch_density", "analytic_train")
    actual.set_mode(train_quantum=False, train_classical=False)
    x, y = torch.randn(3, 6, dtype=torch.float64), torch.tensor([0, 1, 2])
    actual.train()
    with torch.no_grad():
        expected = actual(x)
    with CPUEngine(actual, {"train": (x, y), "val": (x, y)}, CPUConfig(workers=2, microbatch_size=1)) as engine:
        _, training_logits = engine.run("train", [0, 1, 2], backward=False, training=True)
        _, evaluation_logits = engine.run("val", [0, 1, 2], backward=False)
    torch.testing.assert_close(training_logits, expected, atol=1e-12, rtol=1e-12)
    assert (evaluation_logits - expected).abs().max() > 1e-6


def test_unknown_mode_rejected():
    with pytest.raises(ValueError, match="measurement_mode"):
        ModelConfig(6, 3, measurement_mode="typo")
