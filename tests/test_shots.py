from dataclasses import replace

import numpy as np
import pytest
import torch

from qml_beans.model import ModelConfig
from qml_beans.noisy import (NoiseConfig, NoisyHybridClassifier, TorchDensityBlock,
                             sample_expectations)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("shots", [0, -1, True, 1.5])
def test_reject_invalid_shots(shots):
    with pytest.raises(ValueError, match="shots"):
        ModelConfig(8, 2, shots=shots)


def test_joint_measurements_and_sampling_variance():
    probabilities = torch.zeros(2000, 256, dtype=torch.float64)
    # Perfectly correlated first two wires: P(00)=0.7, P(11)=0.3; E[Z]=0.4.
    probabilities[:, 0] = 0.7
    probabilities[:, 192] = 0.3
    shots = 128
    values = sample_expectations(probabilities, shots, torch.Generator().manual_seed(123))
    assert torch.equal(values[:, 0], values[:, 1])
    assert torch.equal(values[:, 2:], torch.ones(2000, 6, dtype=torch.float64))
    assert abs(float(values[:, 0].mean()) - 0.4) < 0.015
    assert float(values[:, 0].var()) == pytest.approx((1 - 0.4**2) / shots, rel=0.12)
    counts = (values + 1) * shots / 2
    assert torch.equal(counts, counts.round())


@pytest.mark.parametrize("backend", ["torch_density", "pennylane_statevector", "pennylane_mixed"])
def test_finite_shot_forward_backward_and_rng_reset(backend):
    config = ModelConfig(8, 2, quantum_depth=1, rotation_gates=("RY",), entanglement="none",
                         shots=256, measurement_seed=123)
    model = NoisyHybridClassifier(config, NoiseConfig(backend=backend, enabled=backend != "pennylane_statevector",
                                                     precision="float64"))
    block = model.quantum
    x = torch.full((2, 8), 0.3, dtype=torch.float64, requires_grad=True)
    block.set_measurement_seed(77)
    a = block(x)
    block.set_measurement_seed(77)
    b = block(x)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert torch.equal((a + 1) * 128, ((a + 1) * 128).round())
    b.square().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert torch.isfinite(block.weights.grad).all() and block.weights.grad.abs().sum() > 0


@pytest.mark.parametrize("embedding,measurement", [("RY", "Z"), ("RZ", "Y")])
def test_gatewise_parameter_shift_handles_reupload_and_pi_factor(monkeypatch, embedding, measurement):
    # Isolate the shift calculus from Monte-Carlo noise. Real sampling is tested separately.
    def exact_measure(probabilities, shots, generator):
        signs = torch.stack([1 - 2 * ((torch.arange(256) >> (7 - wire)) & 1)
                             for wire in range(8)], dim=1).to(probabilities.dtype)
        return probabilities @ signs

    monkeypatch.setattr("qml_beans.noisy.sample_expectations", exact_measure)
    torch.manual_seed(21)
    config = ModelConfig(8, 2, quantum_depth=2, reupload=True, embedding=embedding,
                         measurement=measurement, initial_hadamard=True, rotation_gates=("RX", "RY", "RZ"),
                         entanglement="custom", custom_edges=((0, 7), (2, 4)))
    analytic = TorchDensityBlock(config, NoiseConfig()).double()
    sampled = TorchDensityBlock(replace(config, shots=100), NoiseConfig()).double()
    sampled.load_state_dict(analytic.state_dict())
    x = torch.randn(1, 8, dtype=torch.float64, requires_grad=True)
    upstream = torch.randn(1, 8, dtype=torch.float64)
    a, b = analytic(x), sampled(x)
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    ga = torch.autograd.grad((a * upstream).sum(), (x, analytic.weights))
    gb = torch.autograd.grad((b * upstream).sum(), (x, sampled.weights))
    for left, right in zip(ga, gb):
        torch.testing.assert_close(left, right, atol=2e-10, rtol=2e-10)


def test_measurement_settings_survive_noisy_checkpoint(tmp_path):
    from qml_beans.vision_training import load_noisy_checkpoint

    config = ModelConfig(8, 2, shots=37, measurement_seed=991, quantum_depth=1)
    noise = NoiseConfig()
    model = NoisyHybridClassifier(config, noise)
    path = tmp_path / "shots.pt"
    torch.save(dict(format="noisy_vision_v1", model_config=model.config_dict(), noise_config=model.noise_dict(),
                    state_dict=model.state_dict(), mode=model.mode_dict()), path)
    restored, saved = load_noisy_checkpoint(path)
    assert restored.config.shots == 37 and restored.config.measurement_seed == 991
    assert restored.quantum.config.shots == 37


def test_legacy_analytic_study_without_shot_fields(tmp_path):
    import pandas as pd
    from qml_beans.seeds import StageConfig, collect_seed_runs, make_study_config, run_seed_study

    rng = np.random.default_rng(8)
    frame = pd.DataFrame(rng.normal(size=(40, 3)), columns=['a', 'b', 'c'])
    frame['label'] = ['M', 'R'] * 20
    path = tmp_path / 'data.csv'
    frame.to_csv(path, index=False)
    config = make_study_config(path, ModelConfig(3, 2),
                               [StageConfig(name='classical', quantum=False, train_quantum=False, epochs=1)],
                               target='label')
    config['model_config'].pop('shots')
    config['model_config'].pop('measurement_seed')
    config['model_config'].pop('measurement_mode')
    run_seed_study(config, [3], tmp_path / 'legacy')
    _, rows = collect_seed_runs(tmp_path / 'legacy')
    assert len(rows) == 1
