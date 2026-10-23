from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from threadpoolctl import threadpool_limits

from qml_hybrid.custom_cpp import CustomCppBlock, custom_device_directory, load_custom_engine
from qml_hybrid.gate_errors import CoherentGateErrors, ideal_entangler
from qml_hybrid.measurements import classification_loss
from qml_hybrid.model import ModelConfig
from qml_hybrid.noisy import NoiseConfig, NoisyHybridClassifier, PennyLaneMixedBlock, TorchDensityBlock
from qml_hybrid.vision_data import VisionData
from qml_hybrid.vision_training import CPUConfig, CPUEngine, fit_vision, load_noisy_checkpoint


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        yield
    torch.set_num_threads(previous)


@pytest.fixture
def native_path():
    path = str(custom_device_directory())
    try:
        for kind in ("dm", "sv"):
            load_custom_engine(kind, path)
    except ImportError as exc:
        pytest.skip(f"Optional external CustomDevice build unavailable: {exc}")
    return path


@pytest.mark.parametrize("errors,topology,axis", [(False, "brickwork3", "Z"), (True, "brickwork", "X"),
                                                 (True, "custom", "Y")])
def test_iswap_torch_matches_pennylane(errors, topology, axis):
    config = ModelConfig(8, 3, quantum_depth=2, entangler="iSWAP", entanglement=topology,
                         custom_edges=((7, 0), (1, 5)), measurement=axis,
                         reupload=True, trainable_embedding=True)
    noise = NoiseConfig(precision="float64", gate_errors=errors, t2_us=160)
    a, b = TorchDensityBlock(config, noise).double(), PennyLaneMixedBlock(config, noise).double()
    b.weights.data.copy_(a.weights)
    b.embedding_scale.data.copy_(a.embedding_scale)
    x = torch.randn(1, 8, dtype=torch.float64, requires_grad=True)
    ya, yb = a(x), b(x)
    torch.testing.assert_close(ya, yb, atol=2e-6, rtol=2e-6)
    ga = torch.autograd.grad(ya.square().sum(), (x, a.weights, a.embedding_scale))
    gb = torch.autograd.grad(yb.square().sum(), (x, b.weights, b.embedding_scale))
    for left, right in zip(ga, gb):
        torch.testing.assert_close(left, right, atol=2e-5, rtol=2e-5)
    error = CoherentGateErrors(.99, .9, 42)
    actual = error.entangler("iSWAP", (0, 7))
    relative = actual @ ideal_entangler("iSWAP").conj().T
    fidelity = (abs(np.trace(relative)) ** 2 + 4) / 20
    assert fidelity == pytest.approx(.9)


def test_iswap_drawing_and_cpp_controls(tmp_path):
    import matplotlib.pyplot as plt
    from qml_hybrid.notebook import LayerEditor, draw_vision_circuit
    w = pytest.importorskip("ipywidgets")
    plt.switch_backend("Agg")
    config = ModelConfig(8, 2, quantum_depth=1, entangler="iSWAP", entanglement="brickwork3")
    model = NoisyHybridClassifier(config, NoiseConfig())
    figure, _ = draw_vision_circuit(model, tmp_path / "circuit.png")
    assert (tmp_path / "circuit.png").is_file()
    plt.close(figure)
    root = Path(__file__).parents[1]
    cells = json.loads((root / "notebooks/noisy_vision_cpu.ipynb").read_text(encoding="utf-8"))["cells"]
    source = next("".join(c["source"]) for c in cells if c.get("id") == "model-controls")
    ns = dict(w=w, ROOT=root, os=os, LayerEditor=LayerEditor, display=lambda *args: None)
    exec(source, ns)
    ns["backend"].value = "custom_cpp_sv"
    assert ns["noise_on"].disabled and not ns["noise_on"].value
    assert not ns["custom_device_path"].disabled
    ns["backend"].value = "custom_cpp_dm"
    assert not ns["noise_on"].disabled and ns["noise_on"].value
    ns["entangler"].value = "iSWAP"
    assert ns["entangler"].value == "iSWAP"


def test_notebook_refreshes_gate_errors_before_first_cpp_import():
    # Isolate hot reload from classes imported by the rest of the pytest process.
    root = Path(__file__).parents[1]
    code = '''
import json
from pathlib import Path
import qml_hybrid.gate_errors as gate_errors
del gate_errors.ideal_entangler
notebook = json.loads(Path("notebooks/noisy_vision_cpu.ipynb").read_text(encoding="utf-8"))
source = next("".join(c["source"]) for c in notebook["cells"] if c.get("id") == "create-model")
bootstrap = source.split("plot_run =", 1)[0]
exec(bootstrap, {})
from qml_hybrid.custom_cpp import CustomCppBlock
assert callable(gate_errors.ideal_entangler)
'''
    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("kind,axis,gate,errors,t2", [("dm", "Y", "iSWAP", True, 160),
    ("dm", "X", "CZ", False, 80), ("sv", "Z", "iSWAP", True, 80), ("sv", "Y", "CNOT", False, 80)])
def test_native_reverse_mode_values_and_gradients(native_path, kind, axis, gate, errors, t2):
    torch.manual_seed(13)
    config = ModelConfig(8, 3, quantum_depth=2, embedding="RX", reupload=True,
                         trainable_embedding=True, entangler=gate, entanglement="brickwork3", measurement=axis)
    noise = NoiseConfig(backend="custom_cpp_" + kind, custom_device_path=native_path, enabled=kind == "dm",
                        precision="float64", gate_errors=errors, t2_us=t2, readout_idle_us=.2)
    a = CustomCppBlock(config, noise).double()
    b = TorchDensityBlock(config, replace(noise, backend="torch_density")).double()
    b.load_state_dict(a.state_dict())
    x = torch.randn(1, 8, dtype=torch.float64, requires_grad=True)
    ya, yb = a(x), b(x)
    torch.testing.assert_close(ya, yb, atol=1e-9, rtol=1e-9)
    ga = torch.autograd.grad(ya.square().sum(), (x, a.weights, a.embedding_scale))
    gb = torch.autograd.grad(yb.square().sum(), (x, b.weights, b.embedding_scale))
    for left, right in zip(ga, gb):
        torch.testing.assert_close(left, right, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("kind", ["dm", "sv"])
@pytest.mark.parametrize("mode", ["sampled_backprop", "analytic_train", "reinforce", "parameter_shift"])
def test_native_finite_shots_modes_freezing_and_eval(native_path, kind, mode):
    torch.manual_seed(4)
    config = ModelConfig(6, 3, quantum_depth=1, embedding="RX", entanglement="custom", custom_edges=((0, 7),),
                         entangler="iSWAP", shots=11, measurement_mode=mode, trainable_embedding=True)
    noise = NoiseConfig(backend="custom_cpp_" + kind, custom_device_path=native_path,
                        enabled=kind == "dm", precision="float64", gate_errors=True)
    a = NoisyHybridClassifier(config, noise)
    b = NoisyHybridClassifier(config, replace(noise, backend="torch_density"))
    b.load_state_dict(a.state_dict())
    x, y = torch.randn(1, 6, dtype=torch.float64), torch.tensor([1])
    # Frozen quantum parameters must still transmit the encoder gradient.
    for model in (a, b):
        model.set_mode(train_quantum=False)
        logits = model(x)
        loss = classification_loss(model, logits, y)
        loss.backward()
        assert model.quantum.weights.grad is None
        assert model.encoder[0].weight.grad is not None
        torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(logits, y))
    torch.testing.assert_close(a.encoder[0].weight.grad, b.encoder[0].weight.grad, atol=1e-7, rtol=1e-7)
    with torch.no_grad():
        a.eval()
        a(x)
        assert a.quantum._reinforce_score is None
        a.set_mode(quantum=False)
        assert a(x).shape == (1, 3)


def test_native_gradcheck_and_parallel_accumulation(native_path):
    config = ModelConfig(6, 3, quantum_depth=1, entanglement="none", trainable_embedding=True)
    noise = NoiseConfig(backend="custom_cpp_sv", custom_device_path=native_path, enabled=False, precision="float64")
    a, b = NoisyHybridClassifier(config, noise), NoisyHybridClassifier(config, noise)
    b.load_state_dict(a.state_dict())
    x = torch.randn(1, 8, dtype=torch.float64, requires_grad=True)
    weights = a.quantum.weights.detach().clone().requires_grad_()
    assert torch.autograd.gradcheck(lambda x, w: a.quantum._measure(x, w), (x, weights), eps=1e-6, atol=1e-5)
    data = {"train": (torch.randn(3, 6), torch.tensor([0, 1, 2]))}
    results = []
    for model, workers in ((a, 1), (b, 2)):
        with CPUEngine(model, data, CPUConfig(workers=workers, microbatch_size=1)) as engine:
            results.append(engine.run("train", torch.arange(3), backward=True))
    assert results[0][0] == pytest.approx(results[1][0], abs=1e-10)
    for left, right in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-10, rtol=1e-10)


def test_native_training_checkpoint_and_missing_extension(native_path, tmp_path):
    torch.manual_seed(12)
    config = ModelConfig(6, 2, quantum_depth=1, entanglement="none", trainable_embedding=True,
                         shots=17, measurement_mode="analytic_train")
    noise = NoiseConfig(backend="custom_cpp_sv", custom_device_path=native_path, enabled=False)
    model = NoisyHybridClassifier(config, noise)
    model.set_mode(train_classical=False)
    before = model.quantum.weights.detach().clone()
    data = VisionData(torch.randn(4, 6), torch.tensor([0, 1, 0, 1]), torch.empty(0, 6), torch.empty(0, dtype=torch.long),
                      torch.randn(2, 6), torch.tensor([0, 1]), ("A", "B"), dict(dataset="synthetic"))
    result = fit_vision(model, data, tmp_path / "fit", epochs=1, batch_size=2, cpu=CPUConfig(microbatch_size=1))
    assert not torch.equal(before, model.quantum.weights)
    restored, _ = load_noisy_checkpoint(tmp_path / "fit/best.pt")
    assert isinstance(restored.quantum, CustomCppBlock)
    assert result["metrics"]["test"]["n_samples"] == 2
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name])
    with pytest.raises(ImportError, match="Python"):
        load_custom_engine("dm", str(tmp_path))
