from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t
import torch

from qml_beans.data import download_sonar
from qml_beans.model import ModelConfig
from qml_beans.seeds import (
    StageConfig, _curves, collect_seed_runs, make_study_config, parse_seeds,
    report_seed_study, run_seed_study, sample_statistics,
)
from qml_beans.training import load_checkpoint


@pytest.fixture
def source(tmp_path):
    rng = np.random.default_rng(41)
    frame = pd.DataFrame(rng.normal(size=(90, 5)), columns=list("abcde"))
    frame["Class"] = ["A", "B", "C"] * 30
    path = tmp_path / "sample.csv"
    frame.to_csv(path, index=False)
    return path


def spec(source, **options):
    return make_study_config(source, ModelConfig(5, 3, pre_hidden=8, post_hidden=8, quantum_depth=1),
                              [StageConfig(name="classical", quantum=False, epochs=2)], **options)


def test_statistics_distinguish_sd_sem_and_ci():
    values = np.array([0.70, 0.80, 0.90, 0.85, 0.75])
    stats = sample_statistics(values)
    sd = np.std(values, ddof=1)
    sem = sd / np.sqrt(5)
    half = t.ppf(0.975, 4) * sem
    assert stats["mean"] == pytest.approx(0.8)
    assert stats["sd"] == pytest.approx(sd)
    assert stats["sem"] == pytest.approx(sem)
    assert stats["ci_low"] == pytest.approx(0.8 - half)
    assert stats["ci_high"] == pytest.approx(0.8 + half)
    assert stats["sd"] != stats["sem"]
    one = sample_statistics([0.8])
    assert all(one[name] is None for name in ("sd", "sem", "ci_low", "ci_high"))
    varied = sample_statistics(values, allow_ci=False)
    assert varied["sd"] == pytest.approx(sd)
    assert varied["sem"] is None and varied["ci_low"] is None


@pytest.mark.parametrize("value", ["1,1", "", "-1,2", "x,2", [True], [2**32]])
def test_invalid_seeds_rejected(value):
    with pytest.raises(ValueError):
        parse_seeds(value)


def test_batch_equals_separate_fresh_seed_runs(source, tmp_path):
    config = spec(source)
    batch = run_seed_study(config, [3, 7, 11], tmp_path / "batch")
    separate = tmp_path / "separate"
    for seed in (3, 7, 11):
        run_seed_study(config, [seed], separate)
    _, together = collect_seed_runs([batch])
    _, individual = collect_seed_runs([separate])
    for a, b in zip(together, individual):
        assert a["test_accuracy"] == b["test_accuracy"]
        assert a["test_loss"] == b["test_loss"]
        model_a, checkpoint_a = load_checkpoint(Path(a["output"]) / "best.pt")
        model_b, checkpoint_b = load_checkpoint(Path(b["output"]) / "best.pt")
        for name, parameter in model_a.state_dict().items():
            assert torch.equal(parameter, model_b.state_dict()[name])
        assert checkpoint_a["data"] == checkpoint_b["data"]
    metadata = [json.loads((Path(row["output"]) / "data_metadata.json").read_text()) for row in together]
    assert all(part["indices"] == metadata[0]["indices"] for part in metadata)
    with pytest.raises(ValueError, match="already exists"):
        run_seed_study(config, [3], batch)
    run_seed_study(config, [3], batch, skip_existing=True)
    with pytest.raises(ValueError, match="Duplicate seed"):
        collect_seed_runs([batch, separate])
    report = report_seed_study([batch], tmp_path / "report", band="ci95", show=False)
    assert set(report["summary"]["n"]) == {3}
    assert report["summary"]["ci_low"].notna().all()
    assert (tmp_path / "report" / "metrics_spread.png").is_file()
    assert (tmp_path / "report" / "learning_curves_classical.svg").is_file()
    assert "percentage points (SD), n=3" in report["statements"][0]


def test_separate_directories_pool_and_protocol_mismatch_rejected(source, tmp_path):
    config = spec(source)
    first = run_seed_study(config, [1], tmp_path / "first")
    second = run_seed_study(config, [2], tmp_path / "second")
    _, rows = collect_seed_runs([first, second])
    assert len(rows) == 2
    changed = deepcopy(config)
    changed["stages"][0]["lr"] = 0.01
    with pytest.raises(ValueError, match="different dataset/model/protocol"):
        run_seed_study(changed, [3], first)
    third = run_seed_study(changed, [3], tmp_path / "third")
    with pytest.raises(ValueError, match="Cannot pool"):
        collect_seed_runs([first, third])
    # A copied seed in a second directory is not another independent replicate.
    shutil.copytree(first / "seed_1", tmp_path / "copy")
    with pytest.raises(ValueError, match="Duplicate seed"):
        collect_seed_runs([first, tmp_path / "copy"])
    incomplete = first / "seed_9"
    incomplete.mkdir()
    with pytest.raises(ValueError, match="Incomplete seed"):
        run_seed_study(config, [9], first, skip_existing=True)


def test_varying_splits_omit_ci_and_one_seed_has_undefined_spread(source, tmp_path):
    config = spec(source, split_policy="vary")
    study = run_seed_study(config, [4, 8], tmp_path / "vary")
    _, rows = collect_seed_runs([study])
    meta = [json.loads((Path(row["output"]) / "data_metadata.json").read_text()) for row in rows]
    assert meta[0]["indices"] != meta[1]["indices"]
    result = report_seed_study([study], tmp_path / "report", show=False)
    assert result["summary"]["ci_low"].isna().all()
    assert result["summary"]["sem"].isna().all()
    with pytest.raises(ValueError, match="fixed split"):
        report_seed_study([study], tmp_path / "bad", band="ci95", show=False)
    result = report_seed_study([study / "seed_4"], tmp_path / "one", show=False)
    assert result["summary"]["sd"].isna().all()
    assert "uncertainty across seeds cannot be estimated" in result["statements"][0]


def test_early_stopped_curves_do_not_forward_fill():
    def epoch(index, value):
        return dict(epoch=index, train_loss=value, val_loss=value,
                    train_accuracy=value, val_accuracy=value)
    curves = _curves([dict(seed=1, history=[epoch(1, 0.2)]),
                      dict(seed=2, history=[epoch(1, 0.4), epoch(2, 0.7)])])
    assert curves.n.tolist() == [2, 1]
    assert curves.val_accuracy_mean.tolist() == pytest.approx([0.3, 0.7])
    assert pd.isna(curves.val_accuracy_sd.iloc[1])


def test_staged_seeds_transfer_classical_weights(source, tmp_path):
    config = make_study_config(source, ModelConfig(5, 3, quantum_depth=1), [
        StageConfig(name="classical", quantum=False, epochs=1),
        StageConfig(name="quantum_only", train_classical=False, epochs=1),
        StageConfig(name="joint", epochs=1),
    ])
    directory = run_seed_study(config, [5, 6], tmp_path / "staged")
    _, rows = collect_seed_runs([directory])
    assert len(rows) == 6
    for seed in (5, 6):
        baseline, _ = load_checkpoint(directory / f"seed_{seed}" / "01_classical" / "best.pt")
        quantum, _ = load_checkpoint(directory / f"seed_{seed}" / "02_quantum_only" / "last.pt")
        for name, value in baseline.state_dict().items():
            if not name.startswith("quantum."):
                assert torch.equal(value, quantum.state_dict()[name])


def test_sonar_download_preserves_headerless_first_row_and_caches(monkeypatch, tmp_path):
    raw = pd.DataFrame(np.full((208, 60), 0.125))
    raw[60] = ["M"] * 111 + ["R"] * 97
    payload = raw.to_csv(index=False, header=False).encode()
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: BytesIO(payload))
    path = download_sonar(tmp_path)
    frame = pd.read_csv(path)
    assert frame.shape == (208, 61)
    assert frame.iloc[0, 0] == 0.125
    assert list(frame.columns)[-1] == "Class"
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: pytest.fail("cache must be used"))
    assert download_sonar(tmp_path) == path
