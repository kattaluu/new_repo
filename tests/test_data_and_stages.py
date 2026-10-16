import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.datasets import make_classification

from qml_beans.cli import main
from qml_beans.data import prepare_data, transform_features
from qml_beans.model import HybridClassifier, ModelConfig
from qml_beans.training import fit, load_checkpoint, predict_logits, seed_everything


@pytest.fixture
def dataset(tmp_path):
    x, y = make_classification(n_samples=90, n_features=5, n_informative=4,
                               n_redundant=0, n_classes=3, random_state=12)
    frame = pd.DataFrame(x, columns=[f"x{i}" for i in range(5)])
    frame["label"] = [f"class_{i}" for i in y]
    path = tmp_path / "dataset.csv"
    frame.to_csv(path, index=False)
    return path


def test_preprocessing_uses_only_train_rows_and_restores(dataset):
    data = prepare_data(dataset, target="label")
    metadata = data.metadata
    parts = [set(metadata["indices"][name]) for name in ("train", "val", "test")]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
    assert len(set.union(*parts)) == len(data.y)
    raw = pd.read_csv(dataset)
    train_values = raw.iloc[metadata["indices"]["train"]].drop(columns="label")
    np.testing.assert_allclose(metadata["preprocessing"]["mean"], train_values.mean())
    np.testing.assert_allclose(data.split("train")[0].mean(axis=0), 0, atol=1e-6)
    restored = prepare_data(dataset, saved_metadata=metadata)
    np.testing.assert_array_equal(data.x, restored.x)
    changed = raw.copy()
    changed.loc[0, "x0"] += 1
    changed.to_csv(dataset, index=False)
    with pytest.raises(ValueError, match="differs from checkpoint"):
        prepare_data(dataset, saved_metadata=metadata)


def test_missing_values_and_duplicate_handling(dataset):
    frame = pd.read_csv(dataset)
    frame.loc[0, "x0"] = np.nan
    frame = pd.concat([frame, frame.iloc[[1]]], ignore_index=True)
    frame.to_csv(dataset, index=False)
    data = prepare_data(dataset, target="label")
    assert data.metadata["duplicates_dropped"] == 1
    assert np.isfinite(data.x).all()
    inference = frame.drop(columns="label")
    actual = transform_features(inference[inference.columns[::-1]], data.metadata["preprocessing"])
    assert np.isfinite(actual).all()
    with pytest.raises(ValueError, match="Missing feature"):
        transform_features(inference.drop(columns="x0"), data.metadata["preprocessing"])


def test_classical_to_quantum_checkpoint_and_prediction(dataset, tmp_path):
    seed_everything(4)
    data = prepare_data(dataset, target="label")
    model = HybridClassifier(ModelConfig(5, 3, quantum_depth=1))
    model.set_mode(quantum=False)
    baseline = tmp_path / "classical"
    fit(model, data, baseline, epochs=2, batch_size=16)
    restored, checkpoint = load_checkpoint(baseline / "last.pt")
    torch.testing.assert_close(predict_logits(model, data.x), predict_logits(restored, data.x))
    original_classical = {name: p.detach().clone() for name, p in restored.named_parameters()
                          if not name.startswith("quantum.")}
    restored.set_mode(quantum=True, train_classical=False, train_quantum=True)
    fit(restored, data, tmp_path / "quantum", epochs=1, batch_size=16)
    for name, value in restored.named_parameters():
        if name in original_classical:
            assert torch.equal(value, original_classical[name])
    assert checkpoint["data"]["indices"] == data.metadata["indices"]
    predictions = tmp_path / "predictions.csv"
    main(["predict", "--checkpoint", str(baseline / "best.pt"),
          "--data", str(dataset), "--output", str(predictions)])
    table = pd.read_csv(predictions)
    assert len(table) == 90
    np.testing.assert_allclose(table.filter(like="probability_").sum(axis=1), 1, atol=1e-6)


def test_all_frozen_is_evaluation_only(dataset, tmp_path):
    data = prepare_data(dataset, target="label")
    model = HybridClassifier(ModelConfig(5, 3, quantum_depth=1))
    model.set_mode(quantum=True, train_classical=False, train_quantum=False)
    before = {name: p.clone() for name, p in model.named_parameters()}
    metrics = fit(model, data, tmp_path / "frozen", epochs=1, batch_size=16)
    assert metrics["n_trainable_parameters"] == 0
    for name, value in model.named_parameters():
        assert torch.equal(value, before[name])
