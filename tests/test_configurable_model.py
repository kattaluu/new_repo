from dataclasses import asdict
import json
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
import torch
from torch import nn

from qml_beans.model import HybridClassifier, ModelConfig
from qml_beans.notebook import Experiment, parse_edges, parse_rotations, select_dataset
from qml_beans.training import load_checkpoint, save_checkpoint


@pytest.fixture
def cached_sonar(tmp_path):
    frame = pd.DataFrame(np.random.default_rng(12).random((208, 60)),
                         columns=[f"Attribute{i}" for i in range(1, 61)])
    frame["Class"] = ["M"] * 111 + ["R"] * 97
    source = tmp_path / "data" / "Sonar.csv"
    source.parent.mkdir()
    frame.to_csv(source, index=False)
    return source


@pytest.mark.parametrize("options", [
    dict(embedding="RX", rotation_gates=("RY", "RX", "RZ"), entanglement="linear", entangler="CZ", measurement="Y", reupload=True),
    dict(embedding="RZ", initial_hadamard=True, entanglement="full", measurement="X"),
    dict(entanglement="none", rotation_gates=("RX",), measurement="Z"),
    dict(entanglement="custom", custom_edges=((0, 1), (0, 4), (4, 7)), measurement="Y"),
])
def test_configurable_circuit_batching_and_gradients(options):
    torch.manual_seed(8)
    model = HybridClassifier(ModelConfig(5, 3, pre_layers=(11, 9), pre_activations=("gelu", "silu"),
                                        post_layers=(7,), post_activations=("elu",), **options)).double()
    x = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    prediction = model(x)
    independent = torch.cat([model(sample[None]) for sample in x], dim=0)
    torch.testing.assert_close(prediction, independent, atol=1e-9, rtol=1e-8)
    prediction.square().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert torch.isfinite(model.quantum.weights.grad).all()
    assert model.quantum.weights.grad.abs().sum() > 0


def test_zero_hidden_layers_still_have_two_fully_connected_layers():
    model = HybridClassifier(ModelConfig(6, 4, pre_layers=(), post_layers=(),
                                        pre_activations=(), post_activations=()))
    assert sum(isinstance(module, nn.Linear) for module in model.modules()) == 2
    assert model(torch.randn(2, 6)).shape == (2, 4)


@pytest.mark.parametrize("options", [
    dict(pre_layers=(0,)), dict(pre_layers=(4, 3), pre_activations=("relu",)),
    dict(post_activations=("unknown",)), dict(embedding="RZ"),
    dict(rotation_gates=()), dict(rotation_gates=("H",)),
    dict(entanglement="invalid"), dict(entangler="SWAP"),
    dict(custom_edges=((0, 8),)), dict(custom_edges=((2, 2),)), dict(measurement="Q"),
])
def test_invalid_architecture_rejected(options):
    with pytest.raises(ValueError):
        ModelConfig(5, 3, **options)


def test_config_json_roundtrip_and_checkpoint(tmp_path):
    config = ModelConfig(4, 3, pre_layers=(13, 5), pre_activations=("leaky_relu", "tanh"),
                         post_layers=(), rotation_gates=("RZ", "RY", "RX"),
                         entanglement="custom", custom_edges=((0, 3), (3, 7)), reupload=True)
    restored_config = ModelConfig(**json.loads(json.dumps(asdict(config))))
    assert restored_config == config
    model = HybridClassifier(config)
    model.set_mode(quantum=True, train_classical=False)
    path = tmp_path / "custom.pt"
    save_checkpoint(path, model, {}, [], 0, {})
    restored, _ = load_checkpoint(path)
    x = torch.randn(3, 4)
    torch.testing.assert_close(model(x), restored(x))
    assert restored.config == config
    assert restored.mode_dict() == model.mode_dict()


def test_experiment_stages_transfer_and_restore(tmp_path, cached_sonar):
    source, data = select_dataset(tmp_path, "Sonar (UCI)")
    experiment = Experiment(tmp_path, source, data)
    experiment.build(ModelConfig(60, 2, pre_layers=(12, 6), pre_activations=("gelu", "relu"),
                                 post_layers=(), entanglement="linear", entangler="CZ"))
    baseline = experiment.train_stage(quantum=False, epochs=1, output=tmp_path / "baseline")
    before = {name: value.clone() for name, value in experiment.model.state_dict().items()}
    hybrid = experiment.train_stage(quantum=True, train_classical=False, epochs=1, output=tmp_path / "hybrid")
    for name, value in experiment.model.state_dict().items():
        if not name.startswith("quantum."):
            assert torch.equal(before[name], value)
    assert len(experiment.comparison()) == 2
    experiment.restore(Path(baseline["output"]) / "best.pt")
    assert not experiment.model.use_quantum
    assert hybrid["metrics"]["model_config"]["entangler"] == "CZ"


def test_builtin_dataset_selection(tmp_path, cached_sonar):
    source, data = select_dataset(tmp_path, "Sonar (UCI)")
    assert source.is_file()
    assert data.x.shape == (208, 60)
    assert data.metadata["classes"] == ["M", "R"]
    assert set(data.metadata["indices"]) == {"train", "val", "test"}


def test_custom_selection_and_parsers(tmp_path, cached_sonar):
    source, original = select_dataset(tmp_path, "Sonar (UCI)")
    _, custom = select_dataset(tmp_path, "Свой CSV / ARFF", path=source, target="Class")
    assert custom.metadata == original.metadata
    assert parse_rotations(" ry,rz, RX ") == ("RY", "RZ", "RX")
    assert parse_edges("[[0,1],[3,7]]") == ((0, 1), (3, 7))
    with pytest.raises(ValueError):
        parse_edges("__import__('os')")
    with pytest.raises(ValueError):
        parse_rotations("RX,H")


@pytest.mark.parametrize("name", ["Iris", "Wine", "Breast Cancer"])
def test_removed_dataset_is_not_selectable(tmp_path, name):
    with pytest.raises(ValueError, match="Unknown dataset"):
        select_dataset(tmp_path, name)
