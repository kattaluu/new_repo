from dataclasses import replace
import gzip
import struct

import numpy as np
import pytest
import torch

from qml_beans.model import ModelConfig, QuantumBlock
from qml_beans.noisy import (NoiseConfig, NoisyHybridClassifier, PennyLaneMixedBlock,
                             TorchDensityBlock, thermal_channel)
from qml_beans.vision_data import prepare_vision_arrays, read_idx
from qml_beans.vision_training import CPUConfig, CPUEngine, fit_vision, load_noisy_checkpoint


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("t1,t2", [(100, 80), (100, 160), (100, 200)])
def test_thermal_physics(t1, t2):
    rho = torch.tensor([[[0.5, 0.5], [0.5, 0.5]]], dtype=torch.complex128)
    actual = thermal_channel(rho, 0, 30, t1, t2, n_qubits=1)
    assert actual[0, 1, 1].real == pytest.approx(0.5 * np.exp(-30 / t1))
    assert actual[0, 0, 1].real == pytest.approx(0.5 * np.exp(-30 / t2))
    assert actual[0].trace().real == pytest.approx(1)
    assert torch.linalg.eigvalsh(actual).min() >= -1e-12


@pytest.mark.parametrize("measurement,entangler,t2", [("Z", "CNOT", 80), ("X", "CZ", 160), ("Y", "CNOT", 200)])
def test_backends_outputs_and_gradients(measurement, entangler, t2):
    torch.manual_seed(123)
    config = ModelConfig(8, 2, quantum_depth=1, rotation_gates=("RX", "RY", "RZ"),
                         measurement=measurement, entangler=entangler, initial_hadamard=True,
                         embedding="RZ", entanglement="custom", custom_edges=((0, 7), (3, 4)))
    noise = NoiseConfig(t2_us=t2, single_gate_us=0.2, two_gate_us=0.8, readout_idle_us=0.5)
    custom = TorchDensityBlock(config, noise).double()
    reference = PennyLaneMixedBlock(config, noise).double()
    reference.weights.data.copy_(custom.weights)
    x = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
    a, b = custom(x), reference(x)
    # PennyLane's T2>T1 Kraus construction regularizes zero eigenvalues at pe=0;
    # the resulting ~1e-7 offset is not an error of the direct channel equations.
    tolerance = 1e-6 if t2 > 100 else 2e-8
    torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
    coeffs = torch.linspace(0.1, 1.2, 16, dtype=torch.float64).reshape(2, 8)
    ga = torch.autograd.grad((a * coeffs).sum(), (x, custom.weights))
    gb = torch.autograd.grad((b * coeffs).sum(), (x, reference.weights))
    for left, right in zip(ga, gb):
        torch.testing.assert_close(left, right, atol=1e-5 if t2 > 100 else 2e-7,
                                   rtol=1e-5 if t2 > 100 else 2e-7)


def test_custom_gradient_against_finite_difference():
    config = ModelConfig(8, 2, quantum_depth=1, entanglement="custom", custom_edges=((0, 7),))
    block = TorchDensityBlock(config, NoiseConfig(t2_us=180)).double()
    x = torch.randn(1, 8, dtype=torch.float64, requires_grad=True)
    gradient = torch.autograd.grad(block(x).square().sum(), x)[0][0, 3]
    delta = torch.zeros_like(x)
    delta[0, 3] = 1e-5
    with torch.no_grad():
        numeric = (block(x + delta).square().sum() - block(x - delta).square().sum()) / 2e-5
    torch.testing.assert_close(gradient, numeric, atol=1e-7, rtol=1e-7)


def test_float32_against_double_reference():
    torch.manual_seed(13)
    config = ModelConfig(8, 2, quantum_depth=2, entanglement="ring")
    custom = TorchDensityBlock(config, NoiseConfig())
    reference = PennyLaneMixedBlock(config, NoiseConfig()).double()
    reference.weights.data.copy_(custom.weights)
    x = torch.rand(1, 8, requires_grad=True)
    x64 = x.detach().double().requires_grad_()
    a, b = custom(x), reference(x64)
    torch.testing.assert_close(a.double(), b, atol=2e-5, rtol=2e-5)
    ga = torch.autograd.grad(a.sum(), (x, custom.weights))
    gb = torch.autograd.grad(b.sum(), (x64, reference.weights))
    for left, right in zip(ga, gb):
        torch.testing.assert_close(left.double(), right, atol=5e-5, rtol=5e-5)


def test_no_noise_and_reupload_match_state_vector():
    config = ModelConfig(8, 2, quantum_depth=2, reupload=True, entanglement="linear")
    custom = TorchDensityBlock(config, NoiseConfig(enabled=False)).double()
    ideal = QuantumBlock(2, config).double()
    ideal.weights.data.copy_(custom.weights)
    x = torch.randn(1, 8, dtype=torch.float64)
    torch.testing.assert_close(custom(x), ideal(x), atol=1e-10, rtol=1e-10)


def test_statevector_backend_and_explicit_noise_guard():
    with pytest.raises(ValueError, match="T1/T2"):
        NoiseConfig(backend="pennylane_statevector", enabled=True)
    config = ModelConfig(6, 3, quantum_depth=1, entanglement="linear", embedding="RX")
    ideal = NoisyHybridClassifier(config, NoiseConfig(backend="pennylane_statevector", enabled=False, precision="float64"))
    density = NoisyHybridClassifier(config, NoiseConfig(enabled=False, precision="float64"))
    shared = {key: value for key, value in density.state_dict().items() if key != "quantum.z_signs"}
    ideal.load_state_dict(shared)
    x = torch.randn(2, 6, dtype=torch.float64)
    torch.testing.assert_close(ideal(x), density(x), atol=1e-10, rtol=1e-10)
    ideal.set_mode(train_quantum=False)
    ideal(x).sum().backward()
    assert ideal.encoder[0].weight.grad.abs().sum() > 0
    assert ideal.quantum.weights.grad is None


def test_density_is_physical_and_frozen_quantum_keeps_encoder_gradient():
    model = NoisyHybridClassifier(ModelConfig(6, 10, quantum_depth=1, entanglement="none"), NoiseConfig())
    model.set_mode(train_quantum=False)
    x = torch.randn(2, 6)
    model(x).square().sum().backward()
    assert model.quantum.weights.grad is None
    assert model.encoder[0].weight.grad.abs().sum() > 0
    rho = model.quantum.density_matrix(torch.rand(1, 8)).detach()
    torch.testing.assert_close(rho, rho.mH, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(rho[0].trace().real, torch.tensor(1.0), atol=1e-5, rtol=1e-5)
    assert torch.linalg.eigvalsh(rho).min() > -1e-5


@pytest.mark.parametrize("mode", ["split_batch", "batch_queue"])
def test_parallel_gradient_and_adam_equivalence_unequal_shards(mode):
    torch.manual_seed(4)
    config = ModelConfig(6, 3, quantum_depth=1, entanglement="custom", custom_edges=((0, 1),))
    noise = NoiseConfig(precision="float64")
    a, b = NoisyHybridClassifier(config, noise), NoisyHybridClassifier(config, noise)
    b.load_state_dict(a.state_dict())
    x, y = torch.randn(3, 6), torch.tensor([0, 1, 2])
    data = {"train": (x, y)}
    with CPUEngine(a, data, CPUConfig(microbatch_size=1)) as engine:
        loss_a, logits_a = engine.run("train", [0, 1, 2], backward=True)
    with CPUEngine(b, data, CPUConfig(workers=2, microbatch_size=1, parallel_mode=mode)) as engine:
        loss_b, logits_b = engine.run("train", [0, 1, 2], backward=True)
    assert loss_a == pytest.approx(loss_b, abs=1e-10)
    torch.testing.assert_close(logits_a, logits_b, atol=1e-10, rtol=1e-10)
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa.grad, pb.grad, atol=1e-10, rtol=1e-10)
    torch.optim.Adam(a.parameters(), lr=0.001).step()
    torch.optim.Adam(b.parameters(), lr=0.001).step()
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, atol=1e-10, rtol=1e-10)


def tiny_data(name="MNIST"):
    rng = np.random.default_rng(42)
    channels, size = (3, 32) if name == "CIFAR-10" else (1, 28)
    train = rng.integers(0, 256, (100, channels, size, size), dtype=np.uint8)
    test = rng.integers(0, 256, (40, channels, size, size), dtype=np.uint8)
    return prepare_vision_arrays(train, np.tile(np.arange(10), 10), test, np.tile(np.arange(10), 4),
                                 name=name, image_size=4, val_size=0.2, train_limit=20, val_limit=10, test_limit=10)


@pytest.mark.parametrize("name", ["MNIST", "Fashion MNIST", "CIFAR-10"])
def test_stratified_official_test_and_image_shapes(name):
    data = tiny_data(name)
    assert data.train_x.shape == (20, 48 if name == "CIFAR-10" else 16)
    assert set(data.metadata["indices"]["train"]).isdisjoint(data.metadata["indices"]["val"])
    assert data.test_x.min() >= -1 and data.test_x.max() <= 1
    for y in (data.train_y, data.val_y, data.test_y):
        assert len(torch.unique(y)) == 10


def test_checkpoint_and_noisy_fit(tmp_path):
    data = tiny_data()
    model = NoisyHybridClassifier(ModelConfig(16, 10, quantum_depth=1, entanglement="none"), NoiseConfig())
    result = fit_vision(model, data, tmp_path, epochs=1, batch_size=10, cpu=CPUConfig(microbatch_size=2))
    restored, saved = load_noisy_checkpoint(tmp_path / "best.pt")
    assert saved["noise_config"]["backend"] == "torch_density"
    assert result["metrics"]["test"]["n_samples"] == 10
    with torch.no_grad():
        torch.testing.assert_close(model(data.test_x[:1]), restored(data.test_x[:1]))


def test_read_idx_and_invalid_noise(tmp_path):
    p = tmp_path / "small.gz"
    with gzip.open(p, "wb") as stream:
        stream.write(struct.pack(">II", 2049, 3) + bytes([1, 2, 3]))
    np.testing.assert_array_equal(read_idx(p), [1, 2, 3])
    with pytest.raises(ValueError):
        NoiseConfig(t1_us=10, t2_us=21)
    with pytest.raises(ValueError):
        NoiseConfig(t2_us=[10] * 7)
    with pytest.raises(ValueError):
        CPUConfig(workers=0)


@pytest.mark.parametrize("entangler", ["CNOT", "CZ"])
def test_brickwork_parallel_rows_physics_and_gradients(entangler):
    from qml_beans.noisy import circuit_schedule

    config = ModelConfig(8, 2, quantum_depth=1, entanglement="brickwork", entangler=entangler)
    noise = NoiseConfig(single_gate_us=0.05, two_gate_us=0.3, precision="float64")
    rows = [step for step in circuit_schedule(config, noise) if step[0].endswith("_parallel")]
    assert [row[1] for row in rows] == [((0, 1), (2, 3), (4, 5), (6, 7)), ((1, 2), (3, 4), (5, 6))]
    assert sum(step[2] for step in circuit_schedule(config, noise)) == pytest.approx(0.75)
    custom = TorchDensityBlock(config, noise).double()
    reference = PennyLaneMixedBlock(config, noise).double()
    reference.weights.data.copy_(custom.weights)
    x = torch.randn(1, 8, dtype=torch.float64, requires_grad=True)
    a, b = custom(x), reference(x)
    torch.testing.assert_close(a, b, atol=2e-8, rtol=2e-8)
    ga = torch.autograd.grad(a.square().sum(), (x, custom.weights))
    gb = torch.autograd.grad(b.square().sum(), (x, reference.weights))
    for left, right in zip(ga, gb):
        torch.testing.assert_close(left, right, atol=2e-7, rtol=2e-7)
    ideal = QuantumBlock(1, config).double()
    ideal.weights.data.copy_(custom.weights)
    noiseless = TorchDensityBlock(config, NoiseConfig(enabled=False)).double()
    noiseless.weights.data.copy_(custom.weights)
    torch.testing.assert_close(ideal(x), noiseless(x), atol=1e-10, rtol=1e-10)
