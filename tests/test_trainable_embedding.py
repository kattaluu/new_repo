from dataclasses import replace

import pytest
import torch

from qml_beans.model import ModelConfig
from qml_beans.noisy import NoiseConfig, NoisyHybridClassifier
from qml_beans.vision_training import CPUConfig, CPUEngine, load_noisy_checkpoint


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('embedding', ['RX', 'RY', 'RZ'])
def test_scale_gradients_match_backends_and_finite_difference(embedding):
    config = ModelConfig(8, 2, quantum_depth=2, embedding=embedding, trainable_embedding=True,
                         initial_hadamard=True, reupload=True, measurement='X', entanglement='brickwork3')
    noise = NoiseConfig(gate_errors=True, precision='float64')
    a = NoisyHybridClassifier(config, noise)
    b = NoisyHybridClassifier(config, replace(noise, backend='pennylane_mixed'))
    with torch.no_grad():
        a.quantum.embedding_scale.copy_(torch.linspace(.6, 1.4, 8))
    b.load_state_dict({k: v for k, v in a.state_dict().items() if k != 'quantum.z_signs'})
    x = torch.linspace(-.7, .8, 8, dtype=torch.float64).unsqueeze(0).requires_grad_()
    ya, yb = a.quantum(x), b.quantum(x)
    torch.testing.assert_close(ya, yb, atol=2e-8, rtol=2e-8)
    ga = torch.autograd.grad(ya.square().sum(), (x, a.quantum.weights, a.quantum.embedding_scale))
    gb = torch.autograd.grad(yb.square().sum(), (x, b.quantum.weights, b.quantum.embedding_scale))
    for left, right in zip(ga, gb):
        torch.testing.assert_close(left, right, atol=2e-7, rtol=2e-7)
    assert ga[-1].abs().sum() > 1e-6
    with torch.no_grad():
        a.quantum.embedding_scale[3] += 1e-5
        plus = a.quantum(x).square().sum()
        a.quantum.embedding_scale[3] -= 2e-5
        minus = a.quantum(x).square().sum()
        a.quantum.embedding_scale[3] += 1e-5
    torch.testing.assert_close(ga[-1][3], (plus - minus) / 2e-5, atol=2e-7, rtol=2e-7)
    # Public density-matrix API must apply the same learned scaling.
    rho = a.quantum.density_matrix(x)
    unscaled = NoisyHybridClassifier(replace(config, trainable_embedding=False), noise)
    unscaled.load_state_dict({k: v for k, v in a.state_dict().items() if k != 'quantum.embedding_scale'})
    torch.testing.assert_close(rho, unscaled.quantum.density_matrix(x * a.quantum.embedding_scale))
    ideal = replace(noise, enabled=False)
    density = NoisyHybridClassifier(config, ideal)
    sv = NoisyHybridClassifier(config, replace(ideal, backend='pennylane_statevector'))
    density.load_state_dict(a.state_dict())
    sv.load_state_dict({k: v for k, v in a.state_dict().items() if k != 'quantum.z_signs'})
    torch.testing.assert_close(density(x), sv(x), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('backend', ['torch_density', 'pennylane_mixed', 'pennylane_statevector'])
@pytest.mark.parametrize('mode', ['parameter_shift', 'sampled_backprop', 'analytic_train'])
def test_scale_trains_with_shots_frozen_encoder_and_freezes_with_quantum(backend, mode):
    config = ModelConfig(8, 2, quantum_depth=1, rotation_gates=('RY',), embedding='RX',
                         reupload=True, trainable_embedding=True, entanglement='none',
                         shots=32, measurement_mode=mode)
    model = NoisyHybridClassifier(config, NoiseConfig(backend=backend, enabled=False, gate_errors=True))
    model.set_mode(train_classical=False, train_quantum=True)
    x = torch.randn(2, 8)
    before = model.quantum.embedding_scale.detach().clone()
    model(x).square().sum().backward()
    grad = model.quantum.embedding_scale.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    torch.optim.SGD(model.quantum.parameters(), lr=.01).step()
    assert not torch.equal(before, model.quantum.embedding_scale)
    assert model.encoder[0].weight.grad is None
    model.set_mode(train_classical=True, train_quantum=False)
    model(x).square().sum().backward()
    assert model.quantum.embedding_scale.grad is None
    assert not model.quantum.embedding_scale.requires_grad
    assert model.encoder[0].weight.grad.abs().sum() > 0


def test_finite_shot_scale_parameter_shift_chain_rule(monkeypatch):
    # Remove sampling variance to verify gate-wise shift exactly against backprop,
    # including a non-unit scale and multiple occurrences of the same encoding.
    import qml_beans.noisy as noisy
    def exact_readout(probabilities, shots, rng):
        indices = torch.arange(256)
        signs = torch.stack([1 - 2 * ((indices >> (7 - wire)) & 1) for wire in range(8)], dim=1)
        return probabilities @ signs.to(probabilities)
    monkeypatch.setattr(noisy, 'sample_expectations', exact_readout)
    config = ModelConfig(8, 2, quantum_depth=2, embedding='RX', reupload=True,
                         rotation_gates=('RY',), entanglement='none', trainable_embedding=True)
    noise = NoiseConfig(gate_errors=True, precision='float64')
    analytic = NoisyHybridClassifier(config, noise)
    shifted = NoisyHybridClassifier(replace(config, shots=1), noise)
    with torch.no_grad():
        analytic.quantum.embedding_scale.copy_(torch.linspace(.5, 1.8, 8))
    shifted.load_state_dict(analytic.state_dict())
    x = torch.randn(1, 8, dtype=torch.float64, requires_grad=True)
    coefficient = torch.linspace(.2, 1, 8, dtype=torch.float64)
    for model in (analytic, shifted):
        (model.quantum(x) * coefficient).sum().backward()
    for a, b in zip(analytic.quantum.parameters(), shifted.quantum.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize('enabled', [False, True])
def test_old_and_new_checkpoint_compatibility(enabled, tmp_path):
    config = ModelConfig(8, 2, embedding='RX', trainable_embedding=enabled)
    model = NoisyHybridClassifier(config, NoiseConfig(enabled=False))
    if enabled:
        with torch.no_grad():
            model.quantum.embedding_scale.fill_(1.3)
    saved_config = model.config_dict()
    if not enabled:
        saved_config.pop('trainable_embedding')  # Old checkpoint format.
        assert 'quantum.embedding_scale' not in model.state_dict()
    path = tmp_path/'best.pt'
    torch.save(dict(format='noisy_vision_v1', model_config=saved_config, noise_config=model.noise_dict(),
                    mode=model.mode_dict(), state_dict=model.state_dict()), path)
    restored, _ = load_noisy_checkpoint(path)
    x = torch.randn(2, 8)
    torch.testing.assert_close(restored(x), model(x))
    assert restored.config.trainable_embedding is enabled


def test_scale_cpu_parallel_gradient():
    config = ModelConfig(8, 2, quantum_depth=1, embedding='RX', trainable_embedding=True,
                         entanglement='brickwork3')
    noise = NoiseConfig(gate_errors=True, precision='float64')
    a, b = NoisyHybridClassifier(config, noise), NoisyHybridClassifier(config, noise)
    b.load_state_dict(a.state_dict())
    data = {'train': (torch.randn(3, 8), torch.tensor([0, 1, 0]))}
    with CPUEngine(a, data, CPUConfig(microbatch_size=1)) as engine:
        engine.run('train', [0, 1, 2], backward=True)
    with CPUEngine(b, data, CPUConfig(workers=2, microbatch_size=1)) as engine:
        engine.run('train', [0, 1, 2], backward=True)
    for left, right in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-10, rtol=1e-10)
