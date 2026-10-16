from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from qml_beans.data import audit_splits, class_distribution_table, prepare_cv_data, prepare_data
from qml_beans.model import ModelConfig
from qml_beans.seeds import StageConfig, collect_seed_runs, make_study_config, report_seed_study, run_seed_study
from qml_beans.training import load_checkpoint


@pytest.fixture
def sonar_shape(tmp_path):
    rng = np.random.default_rng(9)
    frame = pd.DataFrame(rng.normal(size=(208, 60)), columns=[f"Attribute{i}" for i in range(1, 61)])
    frame["Class"] = ["R"] * 97 + ["M"] * 111
    path = tmp_path / "sonar.csv"
    frame.to_csv(path, index=False)
    return path


def test_sonar_holdout_preserves_original_class_proportions(sonar_shape):
    data = prepare_data(sonar_shape, seed=42)
    audit = data.metadata["split_audit"]
    counts = class_distribution_table(data).pivot(index="split", columns="label", values="count")
    assert counts.loc["full"].to_dict() == {"M": 111, "R": 97}
    assert counts.loc["train"].to_dict() == {"M": 72, "R": 62}
    assert counts.loc["val"].to_dict() == {"M": 17, "R": 15}
    assert counts.loc["test"].to_dict() == {"M": 22, "R": 20}
    assert audit["disjoint"] and audit["covers_all"] and audit["all_classes_present"]
    assert all(audit["within_one_object"].values())


@pytest.mark.parametrize("seed", [0, 1, 11, 42, 1234])
@pytest.mark.parametrize("k", [2, 3, 5, 10])
def test_sonar_cv_coverage_stratification_and_train_only_scaler(sonar_shape, seed, k):
    folds = prepare_cv_data(sonar_shape, seed=seed, n_splits=k)
    raw = pd.read_csv(sonar_shape).drop(columns="Class").to_numpy()
    observed_test = []
    for index, data in enumerate(folds):
        metadata = data.metadata
        assert metadata["fold"] == index
        assert all(metadata["split_audit"]["within_one_object"].values())
        parts = metadata["indices"]
        observed_test.extend(parts["test"])
        assert len(parts["val"]) == int(np.ceil((208 - len(parts["test"])) * 0.15))
        np.testing.assert_allclose(metadata["preprocessing"]["mean"], raw[parts["train"]].mean(axis=0))
        np.testing.assert_allclose(data.split("train")[0].mean(axis=0), 0, atol=1e-6)
        # Restoring one fold uses its own saved preprocessing/indices, not another split.
        restored = prepare_data(sonar_shape, saved_metadata=metadata)
        np.testing.assert_array_equal(restored.x, data.x)
        assert restored.metadata == metadata
    assert sorted(observed_test) == list(range(208))
    assert not np.array_equal(folds[0].metadata["preprocessing"]["mean"],
                               folds[1].metadata["preprocessing"]["mean"])


def test_invalid_folds_and_partition_leakage_rejected(sonar_shape):
    for k in (0, 1, 98):
        with pytest.raises(ValueError):
            prepare_cv_data(sonar_shape, n_splits=k)
    data = prepare_data(sonar_shape)
    bad = deepcopy(data.metadata["indices"])
    bad["test"][0] = bad["train"][0]
    with pytest.raises(ValueError, match="disjoint"):
        audit_splits(data.y, data.metadata["classes"], bad)


def test_cv_oof_statistics_count_seeds_not_folds_and_validate_artifacts(sonar_shape, tmp_path):
    config = make_study_config(sonar_shape, ModelConfig(60, 2, pre_hidden=8, post_hidden=8, quantum_depth=1),
                                [StageConfig(name="classical", quantum=False, epochs=1)], cv_folds=3)
    study = run_seed_study(config, [3], tmp_path / "cv")
    first = report_seed_study([study], tmp_path / "one", band="ci95", show=False)
    assert set(first["summary"]["n"]) == {1}
    assert first["summary"]["sd"].isna().all() and first["summary"]["ci_low"].isna().all()
    assert set(first["fold_summary"]["n"]) == {3}
    assert first["fold_summary"]["ci_low"].isna().all()
    run_seed_study(config, [7], study)
    _, fold_rows = collect_seed_runs([study])
    result = report_seed_study([study], tmp_path / "two", band="ci95", show=False)
    assert len(fold_rows) == 6 and len(result["rows"]) == 2
    assert set(result["summary"]["n"]) == {2}
    assert result["summary"]["ci_low"].notna().all()
    for seed in (3, 7):
        frame = pd.read_csv(tmp_path / "two" / f"oof_seed_{seed}_classical.csv")
        assert frame.row_index.tolist() == list(range(208))
        accuracy = np.mean(frame.y_true == frame.y_pred)
        macro_f1 = f1_score(frame.y_true, frame.y_pred, average="macro")
        metrics = result["rows"].query("seed == @seed").iloc[0]
        assert metrics.test_accuracy == pytest.approx(accuracy)
        assert metrics.test_macro_f1 == pytest.approx(macro_f1)
        subset = [row for row in fold_rows if row["seed"] == seed]
        weighted = sum(row["n_test"] * row["test_accuracy"] for row in subset) / 208
        assert metrics.test_accuracy == pytest.approx(weighted)
    assert (tmp_path / "two" / "class_distribution.csv").is_file()
    assert (tmp_path / "two" / "available_folds_per_epoch.csv").is_file()
    run_seed_study(config, [3, 7], study, skip_existing=True)
    # Corrupted prediction row IDs cannot masquerade as a completed CV run.
    prediction_file = Path(fold_rows[0]["oof_file"])
    with np.load(prediction_file, allow_pickle=False) as original:
        values = {name: original[name].copy() for name in original.files}
    values["indices"][0] = values["indices"][1]
    np.savez_compressed(prediction_file, **values)
    with pytest.raises(ValueError, match="Invalid out-of-fold"):
        collect_seed_runs([study])


def test_cv_and_holdout_protocols_cannot_mix(sonar_shape, tmp_path):
    model = ModelConfig(60, 2, pre_hidden=8, post_hidden=8, quantum_depth=1)
    stages = [StageConfig(name="classical", quantum=False, epochs=1)]
    cv = make_study_config(sonar_shape, model, stages, cv_folds=2)
    holdout = make_study_config(sonar_shape, model, stages)
    a = run_seed_study(cv, [1], tmp_path / "cv")
    b = run_seed_study(holdout, [2], tmp_path / "holdout")
    with pytest.raises(ValueError, match="Cannot pool"):
        collect_seed_runs([a, b])
    changed = deepcopy(cv)
    changed["cv_folds"] = 3
    changed["test_size"] = 1 / 3
    with pytest.raises(ValueError, match="different dataset/model/protocol"):
        run_seed_study(changed, [4], a)


def test_stages_freeze_classical_weights_inside_each_cv_fold(sonar_shape, tmp_path):
    config = make_study_config(sonar_shape, ModelConfig(60, 2, quantum_depth=1), [
        StageConfig(name="classical", quantum=False, epochs=1),
        StageConfig(name="quantum_only", train_classical=False, epochs=1),
    ], cv_folds=2)
    study = run_seed_study(config, [5], tmp_path / "staged")
    for fold in (1, 2):
        directory = study / "seed_5" / f"fold_{fold:02d}"
        model, _ = load_checkpoint(directory / "01_classical" / "best.pt")
        quantum, _ = load_checkpoint(directory / "02_quantum_only" / "last.pt")
        for name, value in model.state_dict().items():
            if not name.startswith("quantum."):
                assert value.equal(quantum.state_dict()[name])
        assert (directory / "02_quantum_only" / "split_audit.json").is_file()
        assert (directory / "02_quantum_only" / "class_distribution.csv").is_file()


def test_vary_cv_changes_folds_and_omits_seed_ci(sonar_shape, tmp_path):
    config = make_study_config(sonar_shape, ModelConfig(60, 2, pre_hidden=8, post_hidden=8),
                                [StageConfig(name="classical", quantum=False, epochs=1)],
                                cv_folds=3, split_policy="vary")
    study = run_seed_study(config, [4, 8], tmp_path / "vary_cv")
    _, rows = collect_seed_runs([study])
    first = next(row for row in rows if row["seed"] == 4 and row["fold"] == 1)
    second = next(row for row in rows if row["seed"] == 8 and row["fold"] == 1)
    assert first["metadata"]["indices"] != second["metadata"]["indices"]
    result = report_seed_study([study], tmp_path / "report_vary", show=False)
    assert set(result["summary"]["n"]) == {2}
    assert result["summary"]["ci_low"].isna().all()
    with pytest.raises(ValueError, match="fixed split"):
        report_seed_study([study], tmp_path / "invalid", band="ci95", show=False)
