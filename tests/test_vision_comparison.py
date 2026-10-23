from dataclasses import asdict
import json

import numpy as np
import pytest

from qml_hybrid.model import ModelConfig
from qml_hybrid.noisy import NoiseConfig
from qml_hybrid.vision_plots import plot_vision_comparison, comparison_best_metrics
from qml_hybrid.plot_style import PURPLE


def make_study(root, *, quantum=False, evaluation="val", seeds=(11, 22)):
    for seed in seeds:
        path = root / f"seed_{seed}"
        path.mkdir(parents=True)
        history = []
        for epoch in range(1, 4):
            row = dict(epoch=epoch, train_loss=2 - epoch*.2 + seed*.001,
                       train_accuracy=.2 + epoch*.1 + seed*.0001)
            if evaluation:
                row.update({f"{evaluation}_loss": 2 - epoch*.1 + seed*.001,
                            f"{evaluation}_accuracy": .2 + epoch*.05 + seed*.0001})
            history.append(row)
        metrics = dict(model_config=asdict(ModelConfig(2, 2)), noise_config=asdict(NoiseConfig()),
                       mode=dict(quantum=quantum, train_classical=True, train_quantum=True),
                       data=dict(dataset="CIFAR-10", split_seed=42),
                       settings=dict(seed=seed, initialization="independent", epochs=3),
                       validation=dict(accuracy=.5, macro_f1=.5) if evaluation == "val" else None,
                       test=dict(accuracy=.5, macro_f1=.5, loss=1., balanced_accuracy=.5))
        (path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        (path / "history.json").write_text(json.dumps(history), encoding="utf-8")
    return root


def test_comparison_style_means_bands_and_readonly_histories(tmp_path):
    first = make_study(tmp_path / "classical")
    second = make_study(tmp_path / "hybrid", quantum=True)
    before = {path: path.read_bytes() for folder in (first, second) for path in folder.rglob("*.json")}
    result = plot_vision_comparison({"Классическая": first, "Гибридная": second},
                                    tmp_path / "report", max_epochs=2, show=False)
    curves = result["curves"]
    assert curves.epoch.max() == 2
    selected = curves[(curves.series == "Классическая") & (curves.epoch == 1)].iloc[0]
    assert selected.train_loss_mean == pytest.approx(np.mean([1.811, 1.822]))
    assert selected.train_loss_sd == pytest.approx(np.std([1.811, 1.822], ddof=1))
    for axis in result["figure"].axes:
        assert len(axis.lines) == 4
        assert len(axis.collections) == 4
        assert [line.get_linestyle() for line in axis.lines] == ["-", "--", "-", "--"]
        assert not any("seed" in line.get_label() for line in axis.lines)
    left, right = result["figure"].axes
    for axis in (left, right):
        assert axis.texts[0].get_position() == (-0.12, 1.04)
        assert axis.texts[0].get_transform() == axis.transAxes
        assert not axis.texts[0].get_clip_on()
    assert [text.get_text() for text in left.texts] == ["(а)"]
    assert [text.get_text() for text in right.texts] == ["(б)"]
    np.testing.assert_allclose(result["figure"].get_size_inches(), (12, 5))
    assert left.get_legend() is None
    assert right.get_legend() is not None
    assert [line.get_color() for line in left.lines] == ["#008080", "#008080", "#d32f2f", "#d32f2f"]
    for axis in (left, right):
        assert axis.title.get_fontsize() == 16
        assert axis.xaxis.label.get_fontsize() == 16
        assert axis.title.get_fontfamily() == ["serif"]
        assert all(line.get_linewidth() == 2 for line in axis.lines)
    assert [line.get_color() for line in left.lines] == [line.get_color() for line in right.lines]
    assert left.lines[0].get_color() == left.lines[1].get_color()
    assert left.lines[2].get_color() == left.lines[3].get_color()
    assert left.lines[0].get_color() != left.lines[2].get_color()
    for first_band, second_band in zip(left.collections, right.collections):
        np.testing.assert_array_equal(first_band.get_facecolor(), second_band.get_facecolor())
    for suffix in ("png", "svg", "csv"):
        assert (tmp_path / f"report/comparison_curves.{suffix}").is_file()
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("positions", [None, ((-.2, 1.1), (-.08, 1.06))])
def test_panel_label_positions_can_be_changed_or_hidden(tmp_path, positions):
    path = make_study(tmp_path / "study")
    result = plot_vision_comparison({"Model": path}, tmp_path / "out",
                                    panel_positions=positions, show=False)
    for index, axis in enumerate(result["figure"].axes):
        if positions is None:
            assert not axis.texts
        else:
            assert axis.texts[0].get_position() == positions[index]
            assert not axis.texts[0].get_clip_on()


def test_test_curves_are_labelled_test_and_single_seed_has_no_band(tmp_path):
    path = make_study(tmp_path / "test", evaluation="test")
    result = plot_vision_comparison({"Серия": path}, tmp_path / "out", seeds=[22], band="ci95", show=False)
    axis = result["figure"].axes[0]
    assert axis.lines[1].get_label() == "Тест"
    assert not axis.collections
    manifest = json.loads((tmp_path / "out/comparison.json").read_text(encoding="utf-8"))
    assert manifest["series"][0]["seeds"] == [22]


def test_old_final_test_does_not_become_epoch_curve(tmp_path):
    path = make_study(tmp_path / "old", evaluation=None)
    result = plot_vision_comparison({"Старая серия": path}, tmp_path / "auto", show=False)
    assert len(result["figure"].axes[0].lines) == 1
    assert result["notes"]
    with pytest.raises(ValueError, match="history.json"):
        plot_vision_comparison({"Серия": path}, tmp_path / "explicit", eval_split="test", show=False)


def test_unfinished_history_uses_checkpoint_metadata_without_final_metrics(tmp_path):
    import torch
    folder = make_study(tmp_path / "unfinished", seeds=(11,))
    seed = folder / "seed_11"
    metrics_file = seed / "metrics.json"
    metadata = json.loads(metrics_file.read_text(encoding="utf-8"))
    saved = {key: metadata[key] for key in ("model_config", "noise_config", "mode", "settings", "data")}
    saved.update(format="noisy_vision_v1", epoch=1)
    torch.save(saved, seed / "best.pt")
    metrics_file.unlink()
    before = {p: p.read_bytes() for p in seed.iterdir()}
    result = plot_vision_comparison({"Unfinished": folder}, tmp_path / "plots", show=False)
    assert result["curves"].epoch.tolist() == [1, 2, 3]
    assert len(result["figure"].axes[0].lines) == 2
    assert "test_loss_mean" not in result["curves"]
    assert not metrics_file.exists()
    assert all(p.read_bytes() == content for p, content in before.items())


def test_mixed_protocols_inside_a_group_are_rejected(tmp_path):
    path = make_study(tmp_path / "mixed")
    file = path / "seed_22/metrics.json"
    metadata = json.loads(file.read_text(encoding="utf-8"))
    metadata["model_config"]["quantum_depth"] = 4
    file.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="протокол"):
        plot_vision_comparison({"Ошибочная группа": path}, tmp_path / "out", show=False)


def test_validation_and_test_are_not_mixed_in_one_figure(tmp_path):
    validation = make_study(tmp_path / "val", evaluation="val")
    test = make_study(tmp_path / "test", evaluation="test")
    with pytest.raises(ValueError, match="отдельных"):
        plot_vision_comparison({"Val": validation, "Test": test}, tmp_path / "out", show=False)


def test_xlim_changes_both_axes_without_filtering_data(tmp_path):
    path = make_study(tmp_path / "study")
    result = plot_vision_comparison({"Model": path}, tmp_path / "out", xlim=(0, 2),
                                    ylim_loss=(0, 2.5), ylim_accuracy=(0.1, 0.9), show=False)
    assert result["curves"].epoch.tolist() == [1, 2, 3]
    assert result["figure"].axes[0].get_ylim() == pytest.approx((0, 2.5))
    assert result["figure"].axes[1].get_ylim() == pytest.approx((0.1, 0.9))
    for axis in result["figure"].axes:
        assert axis.get_xlim() == pytest.approx((0, 2))
        assert axis.lines[0].get_xdata().tolist() == [1, 2, 3]
    manifest = json.loads((tmp_path / "out/comparison.json").read_text(encoding="utf-8"))
    assert manifest["xlim"] == [0, 2]
    assert manifest["ylim_loss"] == [0, 2.5]
    assert manifest["ylim_accuracy"] == [0.1, 0.9]
    import pandas as pd
    assert pd.read_csv(tmp_path / "out/comparison_curves.csv").epoch.tolist() == [1, 2, 3]


@pytest.mark.parametrize("limits", [40, (0,), (0, 1, 2), (2, 2), (3, 1), (0, float("inf")), (float("nan"), 2), ("bad", 2)])
def test_invalid_xlim_is_rejected(tmp_path, limits):
    with pytest.raises(ValueError, match="xlim"):
        plot_vision_comparison({"Model": tmp_path}, tmp_path / "out", xlim=limits, show=False)


@pytest.mark.parametrize("name", ["ylim_loss", "ylim_accuracy"])
@pytest.mark.parametrize("limits", [2, (2, 1), (0, float("nan"))])
def test_invalid_ylim_is_rejected(tmp_path, name, limits):
    with pytest.raises(ValueError, match=name):
        plot_vision_comparison({"Model": tmp_path}, tmp_path / "out", **{name: limits}, show=False)


@pytest.mark.parametrize("band", ["sd", "ci95"])
def test_test_reference_uses_last_visible_epoch_and_selected_seeds(tmp_path, band):
    path = make_study(tmp_path / "study", evaluation="test", seeds=(11, 22, 33))
    result = plot_vision_comparison({"Model": path}, tmp_path / "out", band=band,
                                    seeds=[11, 22], xlim=(0, 2.5), test_reference="Model", show=False)
    reference = result["test_reference"]
    assert reference["epoch"] == 2
    assert result["curves"].epoch.max() == 3
    row = result["curves"].query("epoch == 2").iloc[0]
    for axis, metric in zip(result["figure"].axes, ("loss", "accuracy")):
        values = reference["metrics"][metric]
        assert values["mean"] == pytest.approx(row[f"test_{metric}_mean"])
        expected_low = (row[f"test_{metric}_mean"] - row[f"test_{metric}_sd"] if band == "sd"
                        else row[f"test_{metric}_ci_low"])
        assert values["low"] == pytest.approx(expected_low)
        assert axis.lines[-1].get_color() == PURPLE
        assert axis.lines[-1].get_linestyle() == "--"
        assert axis.lines[-1].get_label() == "_nolegend_"
        assert list(axis.lines[-1].get_ydata()) == pytest.approx([values["mean"]]*2)
        assert len(axis.patches) == 1
    manifest = json.loads((tmp_path / "out/comparison.json").read_text(encoding="utf-8"))
    for key in ("series", "epoch", "band", "metrics", "seeds", "paths"):
        assert manifest["test_reference"][key] == reference[key]
    assert "protocol" in manifest["test_reference"]


@pytest.mark.parametrize("external", [False, True])
def test_explicit_reference_epoch_overrides_visible_range(tmp_path, external):
    path = make_study(tmp_path / "study", evaluation="test")
    reference = make_study(tmp_path / "external", evaluation="test") if external else "Model"
    result = plot_vision_comparison({"Model": path}, tmp_path / "out", test_reference=reference,
                                    test_epoch=3, xlim=(0, 1.5), max_epochs=1, show=False)
    assert result["curves"].epoch.max() == 1
    assert result["test_reference"]["epoch"] == 3
    assert result["test_reference"]["metrics"]["loss"]["mean"] == pytest.approx(1.7165)
    assert result["figure"].axes[0].get_xlim() == pytest.approx((0, 1.5))


@pytest.mark.parametrize("epoch", [0, -1, 1.5, True, "2"])
def test_invalid_reference_epoch_is_rejected(tmp_path, epoch):
    with pytest.raises(ValueError, match="test_epoch"):
        plot_vision_comparison({"Model": tmp_path}, tmp_path / "out", test_epoch=epoch, show=False)


def test_missing_exact_reference_epoch_is_rejected(tmp_path):
    path = make_study(tmp_path / "study", evaluation="test")
    with pytest.raises(ValueError, match="10"):
        plot_vision_comparison({"Model": path}, tmp_path / "out", test_reference="Model", test_epoch=10, show=False)


def test_reference_single_seed_has_line_without_band(tmp_path):
    path = make_study(tmp_path / "study", evaluation="test", seeds=(11,))
    result = plot_vision_comparison({"Model": path}, tmp_path / "out", test_reference="Model", show=False)
    assert result["test_reference"]["epoch"] == 3
    for axis in result["figure"].axes:
        assert len(axis.lines) == 3
        assert not axis.patches


def test_reference_requires_historical_test_inside_range(tmp_path):
    path = make_study(tmp_path / "study", evaluation="val")
    with pytest.raises(ValueError, match="history.json"):
        plot_vision_comparison({"Model": path}, tmp_path / "out", test_reference="Model", show=False)
    path = make_study(tmp_path / "test", evaluation="test")
    with pytest.raises(ValueError, match="xlim"):
        plot_vision_comparison({"Model": path}, tmp_path / "out", test_reference="Model", xlim=(4, 5), show=False)
    with pytest.raises(ValueError, match="test_reference"):
        plot_vision_comparison({"Model": path}, tmp_path / "out", test_reference="Unknown", show=False)


@pytest.mark.parametrize("as_list", [False, True])
def test_independent_reference_is_not_added_to_comparison_curves(tmp_path, as_list):
    plotted = make_study(tmp_path / "plotted", evaluation="val")
    external = make_study(tmp_path / "external", quantum=True, evaluation="test", seeds=(11, 22, 33))
    paths = [external / f"seed_{seed}" for seed in (11, 22, 33)] if as_list else str(external)
    result = plot_vision_comparison({"Model": plotted}, tmp_path / "out", test_reference=paths,
                                    reference_label="External", reference_seeds=[22, 33],
                                    seeds=[11], xlim=(0, 2.5), show=False)
    reference = result["test_reference"]
    assert reference["series"] == "External"
    assert reference["epoch"] == 2
    assert reference["seeds"] == [22, 33]
    assert len(reference["paths"]) == 2
    assert reference["metrics"]["loss"]["mean"] == pytest.approx(1.8 + .0275)
    assert reference["metrics"]["loss"]["low"] is not None
    assert set(result["curves"].series) == {"Model"}
    for axis in result["figure"].axes:
        assert len(axis.lines) == 3
        assert axis.lines[-1].get_label() == "External"
        assert len(axis.patches) == 1


@pytest.mark.parametrize("band", ["sd", "ci95"])
def test_best_metrics_use_visible_means_and_reference_epoch(tmp_path, band):
    path = make_study(tmp_path / "study", evaluation="test")
    result = plot_vision_comparison({"Model": path}, tmp_path / "out", xlim=(0, 2.5), band=band,
                                    test_reference="Model", test_epoch=3, show=False)
    table = comparison_best_metrics(result)
    assert len(table) == 8
    best = table[table["Критерий"] != "Заданная эпоха ориентира"]
    assert best["Эпоха"].tolist() == [2]*4
    np.testing.assert_allclose(best["Значение"], [1.6165, .40165, 1.8165, .30165])
    assert (best["Нижняя граница"] < best["Значение"]).all()
    assert (best["Верхняя граница"] > best["Значение"]).all()
    np.testing.assert_allclose(table["Величина разброса (±)"],
                               (table["Верхняя граница"] - table["Нижняя граница"]) / 2)
    assert table.iloc[-1]["Эпоха"] == 3
    assert table.iloc[-1]["Число запусков"] == 2
    reference_train = table[table["Выборка"] == "Обучающая — ориентир"]
    assert reference_train["Эпоха"].tolist() == [3, 3]
    np.testing.assert_allclose(reference_train["Значение"], [1.4165, .50165])
    assert (reference_train["Величина разброса (±)"] > 0).all()


def test_best_metrics_do_not_substitute_validation_for_test(tmp_path, capsys):
    path = make_study(tmp_path / "study", evaluation="val", seeds=(11,))
    result = plot_vision_comparison({"Model": path}, tmp_path / "out", show=False)
    table = comparison_best_metrics(result)
    assert len(table) == 2
    assert table["Выборка"].tolist() == ["Обучающая"]*2
    assert table["Нижняя граница"].isna().all()
    assert table["Величина разброса (±)"].isna().all()
    assert "нет test" in capsys.readouterr().out


@pytest.mark.parametrize("legend_loc", ["lower right", "upper left", None])
def test_vision_seed_report_uses_same_style_and_purple(tmp_path, monkeypatch, legend_loc):
    from matplotlib.figure import Figure
    from qml_hybrid.vision_training import report_vision_study
    path = make_study(tmp_path / "study", evaluation="test")
    captured = {}
    def capture(figure, filename, **kwargs):
        captured[filename.name] = figure
    monkeypatch.setattr(Figure, "savefig", capture)
    report_vision_study(sorted(path.glob("seed_*")), tmp_path / "out", show=False,
                         legend_loc=legend_loc, legend_ncol=2)
    figure = captured["learning_curves_classical.png"]
    np.testing.assert_allclose(figure.get_size_inches(), (12, 5))
    assert figure.axes[0].get_legend() is None
    if legend_loc is None:
        assert figure.axes[1].get_legend() is None
    else:
        assert figure.axes[1].get_legend()._loc == {"lower right": 4, "upper left": 2}[legend_loc]
        assert figure.axes[1].get_legend()._ncols == 2
    for axis in figure.axes:
        assert [line.get_color() for line in axis.lines] == [PURPLE, PURPLE]
        assert [line.get_linestyle() for line in axis.lines] == ["-", "--"]
        assert all(line.get_marker() == "None" for line in axis.lines)
        assert axis.title.get_fontsize() == 16
        assert axis.title.get_fontfamily() == ["serif"]


@pytest.mark.parametrize("legend_loc", ["upper left", None])
@pytest.mark.parametrize("legend_on", ["left", "right", "both", "none"])
def test_comparison_legend_position_and_hide(tmp_path, legend_loc, legend_on):
    path = make_study(tmp_path / "study")
    result = plot_vision_comparison({"Model": path}, tmp_path / "out", show=False,
                                    legend_loc=legend_loc, legend_on=legend_on)
    left, right = result["figure"].axes
    for axis, side in ((left, "left"), (right, "right")):
        if legend_loc is not None and legend_on in (side, "both"):
            assert axis.get_legend()._loc == 2
        else:
            assert axis.get_legend() is None


def test_invalid_legend_side_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="legend_on"):
        plot_vision_comparison({"Model": tmp_path}, tmp_path / "out", legend_on="invalid", show=False)


def test_training_only_has_series_caption_in_legend(tmp_path):
    first = make_study(tmp_path / "first", evaluation="test")
    second = make_study(tmp_path / "second", evaluation="test", quantum=True)
    result = plot_vision_comparison({"3.000 изображений": first, "10.000 изображений": second},
                                    tmp_path / "out", show=False, legend_ncol=2)
    assert result["figure"].axes[1].get_legend()._ncols == 2
    for axis in result["figure"].axes:
        assert [line.get_label() for line in axis.lines] == [
            "Обучение (3.000 изображений)", "Тест",
            "Обучение (10.000 изображений)", "Тест"]


@pytest.mark.parametrize("columns", [0, -1, 1.5, True])
def test_invalid_legend_columns_are_rejected(tmp_path, columns):
    with pytest.raises(ValueError, match="legend_ncol"):
        plot_vision_comparison({"Model": tmp_path}, tmp_path / "out", legend_ncol=columns, show=False)


def test_mixed_report_preserves_protocols_and_uses_available_epochs(tmp_path):
    from qml_hybrid.vision_training import report_vision_study
    import pandas as pd
    path = make_study(tmp_path / "copy", evaluation="test")
    file = path / "seed_22/metrics.json"
    metadata = json.loads(file.read_text(encoding="utf-8"))
    metadata["model_config"]["quantum_depth"] = 4
    metadata["settings"]["epochs"] = 2
    file.write_text(json.dumps(metadata), encoding="utf-8")
    history = path / "seed_22/history.json"
    history.write_text(json.dumps(json.loads(history.read_text(encoding="utf-8"))[:2]), encoding="utf-8")
    before = {p: p.read_bytes() for p in path.rglob("*.json")}
    paths = sorted(path.glob("seed_*"))
    with pytest.raises(ValueError, match="model.quantum_depth"):
        report_vision_study(paths, tmp_path / "strict", show=False)
    summary = report_vision_study(paths, tmp_path / "mixed", show=False, allow_mixed_protocols=True)
    assert summary.ci_low.isna().all()
    manifest = json.loads((tmp_path / "mixed/summary.json").read_text(encoding="utf-8"))
    assert manifest["mixed_protocols"] is True
    assert len(manifest["protocol"]["runs"]) == 2
    assert manifest["protocol"]["runs"][1]["protocol"]["model"]["quantum_depth"] == 4
    curves = pd.read_csv(tmp_path / "mixed/curves_classical.csv")
    assert curves.n.tolist() == [2, 2, 1]
    assert curves.train_loss_ci_low.isna().all()
    assert all(p.read_bytes() == content for p, content in before.items())
    with pytest.raises(ValueError, match="band='sd'"):
        report_vision_study(paths, tmp_path / "ci", band="ci95", show=False, allow_mixed_protocols=True)


def test_matching_protocols_from_different_folders_and_duplicate_diagnostic(tmp_path):
    from qml_hybrid.vision_training import collect_vision_runs
    a = make_study(tmp_path / "timestamp_a", seeds=(11,)) / "seed_11"
    b = make_study(tmp_path / "timestamp_b", seeds=(22,)) / "seed_22"
    rows, protocol, seeds = collect_vision_runs([a, b])
    assert seeds == [11, 22]
    assert not protocol.get("mixed_protocols")
    with pytest.raises(ValueError, match="Повторный seed=11"):
        collect_vision_runs([a, a], allow_mixed_protocols=True)


def test_comparison_allows_mixed_protocols_only_for_selected_series(tmp_path):
    first = make_study(tmp_path / "ordinary", evaluation="test")
    second = make_study(tmp_path / "mixed", evaluation="test")
    file = second / "seed_22/metrics.json"
    metadata = json.loads(file.read_text(encoding="utf-8"))
    metadata["settings"]["batch_size"] = 10000
    file.write_text(json.dumps(metadata), encoding="utf-8")
    studies = {"ordinary": first, "mixed": second}
    with pytest.raises(ValueError, match="settings.batch_size"):
        plot_vision_comparison(studies, tmp_path / "strict", show=False)
    result = plot_vision_comparison(studies, tmp_path / "out", show=False,
                                    allow_mixed_protocols={"mixed": True})
    mixed = result["curves"].query("series == 'mixed'")
    assert mixed.train_loss_ci_low.isna().all()
    assert result["curves"].query("series == 'ordinary'").train_loss_ci_low.notna().all()
    manifest = json.loads((tmp_path / "out/comparison.json").read_text(encoding="utf-8"))
    assert manifest["mixed_protocols"] is True
    assert manifest["series"][1]["protocol"]["mixed_protocols"] is True
    assert result["figure"]._suptitle is None
    with pytest.raises(ValueError, match="band='sd'"):
        plot_vision_comparison(studies, tmp_path / "ci", show=False, band="ci95",
                               allow_mixed_protocols={"mixed": True})
