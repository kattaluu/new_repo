"""Publication-style comparisons of already saved vision learning histories."""

from pathlib import Path

import numpy as np
import pandas as pd

from .seeds import _curves
from .training import write_json
from .plot_style import PURPLE, vision_rc, validate_panel_positions, add_panel_label


def _run_paths(paths):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    result = []
    for path in paths:
        path = Path(path)
        if path.name == "metrics.json" and path.is_file():
            result.append(path.parent)
        elif (path / "metrics.json").is_file():
            result.append(path)
        else:
            found = sorted(p.parent for p in path.glob("seed_*/metrics.json"))
            if not found:
                raise ValueError(f"Не найдены сохранённые metrics.json в {path}")
            result.extend(found)
    return result


def plot_vision_comparison(studies, output, *, band="sd", seeds=None, eval_split="auto",
                           max_epochs=None, xlim=None, ylim_loss=None, ylim_accuracy=None,
                           test_reference=None, test_epoch=None, reference_seeds=None, reference_label=None,
                           show=True, font="Times New Roman", legend_loc="lower right", legend_on="right",
                           legend_ncol=1, allow_mixed_protocols=False,
                           panel_positions=((-0.12, 1.04), (-0.12, 1.04))):
    """Solid train/dashed validation or test, with SD/CI across selected seeds.

    studies maps legend labels to study folders, seed folders or lists of folders.
    Protocols may differ between labels, but must agree inside each seed group.
    Missing historical test curves are never reconstructed from final test metrics.
    xlim=(left, right) changes the visible epoch range without filtering curve data.
    ylim_loss and ylim_accuracy independently set the visible vertical ranges.
    test_reference accepts a studies label or independent study/seed folder(s).
    reference_seeds selects seeds for an independent reference; None uses all.
    A horizontal test mean/band is drawn at the
    last recorded epoch inside xlim (and max_epochs), independently of eval_split.
    test_epoch selects an exact recorded reference epoch, independently of xlim
    and max_epochs; None retains the last-visible-epoch selection.
    """
    from .vision_training import collect_vision_runs
    import matplotlib as mpl
    import matplotlib.style as mpl_style
    from matplotlib.ticker import MaxNLocator

    if not studies or not isinstance(studies, dict):
        raise ValueError("studies: непустой словарь {подпись: папка или список папок}")
    if band not in ("sd", "ci95") or eval_split not in ("auto", "val", "test"):
        raise ValueError("band: sd/ci95; eval_split: auto/val/test")
    if legend_on not in ("left", "right", "both", "none"):
        raise ValueError("legend_on: left / right / both / none")
    if type(legend_ncol) is not int or legend_ncol < 1:
        raise ValueError("legend_ncol: положительное целое")
    panel_positions = validate_panel_positions(panel_positions)
    if max_epochs is not None and (type(max_epochs) is not int or max_epochs < 1):
        raise ValueError("max_epochs: положительное целое или None")
    def validate_limits(value, name):
        if value is None:
            return None
        try:
            limits = np.asarray(value, dtype=float)
            valid = limits.shape == (2,) and np.isfinite(limits).all() and limits[0] < limits[1]
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(f"{name}: (lower, upper), finite values with lower < upper")
        return limits.tolist()

    xlim = validate_limits(xlim, "xlim")
    ylim_loss = validate_limits(ylim_loss, "ylim_loss")
    ylim_accuracy = validate_limits(ylim_accuracy, "ylim_accuracy")
    if test_epoch is not None and (type(test_epoch) is not int or test_epoch < 1):
        raise ValueError("test_epoch: положительное целое или None")
    prepared, frames, notes = [], [], []
    def mixed_allowed(label):
        return allow_mixed_protocols.get(label, False) if isinstance(allow_mixed_protocols, dict) else allow_mixed_protocols

    def read_group(paths, selected, label):
        rows, protocol, chosen = collect_vision_runs(_run_paths(paths), seeds=selected,
                                                    allow_mixed_protocols=mixed_allowed(label))
        mixed = protocol.get("mixed_protocols", False)
        if mixed and band != "sd":
            raise ValueError(f"{label}: для смешанной серии используйте band='sd'")
        if mixed:
            notes.append(f"{label}: смешанная серия; SD включает различия настроек. "
                         "На каждой эпохе используются доступные запуски.")
        return rows, protocol, chosen

    for label, paths in studies.items():
        selected = seeds.get(label) if isinstance(seeds, dict) else seeds
        rows, protocol, chosen = read_group(paths, selected, label)
        curves = _curves(rows, allow_ci=not protocol.get("mixed_protocols", False))
        reference_curves = curves
        if curves.empty:
            raise ValueError(f"{label}: история обучения пуста")
        if max_epochs is not None:
            curves = curves[curves.epoch <= max_epochs].copy()
        if curves.empty:
            raise ValueError(f"{label}: нет эпох в выбранном диапазоне")
        available = [split for split in ("val", "test")
                     if all(f"{split}_{metric}_mean" in curves for metric in ("loss", "accuracy"))]
        evaluation = next(iter(available), None) if eval_split == "auto" else eval_split
        if evaluation is not None and evaluation not in available:
            raise ValueError(f"{label}: в history.json нет {evaluation} loss/accuracy по эпохам. "
                             "Итоговых тестовых метрик недостаточно для восстановления кривой.")
        if evaluation is None:
            notes.append(f"{label}: в старой истории нет val/test по эпохам; показана только обучающая кривая.")
        if len(chosen) == 1:
            notes.append(f"{label}: один seed, разброс не определён; полоса не строится.")
        frame = curves.copy()
        frame.insert(0, "series", label)
        frames.append(frame)
        prepared.append(dict(label=label, curves=curves, reference_curves=reference_curves,
                             evaluation=evaluation, seeds=chosen,
                             paths=[row["output"] for row in rows], protocol=protocol))
    if len({group["evaluation"] for group in prepared if group["evaluation"] is not None}) > 1:
        raise ValueError("Val и test должны быть на отдельных рисунках. Выберите общую выборку "
                         "через eval_split='val' или 'test' и серии с этими историями.")
    reference = None
    if test_reference is not None:
        if isinstance(test_reference, str) and test_reference in studies:
            group = next(group for group in prepared if group["label"] == test_reference)
            if reference_seeds is not None:
                rows, protocol, chosen = read_group(studies[test_reference], reference_seeds, test_reference)
                group = dict(label=test_reference, curves=_curves(rows, allow_ci=not protocol.get("mixed_protocols", False)), seeds=chosen,
                             paths=[row["output"] for row in rows], protocol=protocol)
        else:
            try:
                rows, protocol, chosen = read_group(test_reference, reference_seeds, reference_label)
            except ValueError as exc:
                raise ValueError(f"test_reference: {exc}") from exc
            group = dict(label=reference_label or "Отдельное измерение",
                         curves=_curves(rows, allow_ci=not protocol.get("mixed_protocols", False)), seeds=chosen,
                         paths=[row["output"] for row in rows], protocol=protocol)
        label = reference_label or group["label"]
        candidates = group.get("reference_curves", group["curves"])
        if test_epoch is None and max_epochs is not None:
            candidates = candidates[candidates.epoch <= max_epochs]
        required = [f"test_{metric}_mean" for metric in ("loss", "accuracy")]
        if not all(key in candidates for key in required):
            raise ValueError(f"{label}: для ориентира нужны test loss/accuracy по эпохам "
                             "в history.json; итоговый test не заменяет значение на выбранной эпохе.")
        if test_epoch is not None:
            candidates = candidates[candidates.epoch == test_epoch]
        elif xlim is not None:
            candidates = candidates[candidates.epoch.between(*xlim)]
        candidates = candidates[np.isfinite(candidates[required]).all(axis=1)]
        if candidates.empty:
            location = f"на эпохе {test_epoch}" if test_epoch is not None else "в пределах xlim"
            raise ValueError(f"{label}: нет тестовых значений {location}")
        row = candidates.sort_values("epoch").iloc[-1]
        reference = dict(series=label, epoch=int(row.epoch), band=band, n=int(row.n), metrics={}, train_metrics={},
                         seeds=group["seeds"], paths=group["paths"], protocol=group["protocol"])
        if len(group["seeds"]) == 1:
            note = f"{label}: один seed у тестового ориентира; полоса разброса не строится."
            if note not in notes:
                notes.append(note)
        for key in ("test_loss", "test_accuracy", "train_loss", "train_accuracy"):
            split, metric = key.split("_", 1)
            if f"{key}_mean" not in row or not pd.notna(row[f"{key}_mean"]):
                notes.append(f"{label}: нет {key} на эпохе {int(row.epoch)} для таблицы ориентира.")
                continue
            mean = float(row[f"{key}_mean"])
            if row.n < 2:
                low, high = np.nan, np.nan
            elif band == "sd":
                delta = float(row[f"{key}_sd"])
                low, high = mean - delta, mean + delta
            else:
                low, high = float(row[f"{key}_ci_low"]), float(row[f"{key}_ci_high"])
            has_band = row.n >= 2 and np.isfinite([low, high]).all()
            destination = "metrics" if split == "test" else "train_metrics"
            reference[destination][metric] = dict(mean=mean, low=low if has_band else None,
                                                  high=high if has_band else None)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    colors = ("#008080", "#d32f2f", "#4c72b0", "#dd8452", "#55a868", "#8172b3")
    split_names = dict(train="Обучение", val="Валидационная выборка", test="Тест")
    with mpl_style.context("seaborn-v0_8-whitegrid"), mpl.rc_context(vision_rc(font)):
        if show:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        else:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            fig = Figure(figsize=(12, 5))
            FigureCanvasAgg(fig)
            axes = fig.subplots(1, 2)
        for axis, metric, title, ylabel in (
            (axes[0], "loss", "Функция потерь (Loss)", "Значение кросс-энтропии"),
            (axes[1], "accuracy", "Точность (Accuracy)", "Значение точности"),
        ):
            for index, group in enumerate(prepared):
                curves = group["curves"]
                color = colors[index % len(colors)]
                splits = ["train"] + ([group["evaluation"]] if group["evaluation"] else [])
                for split in splits:
                    key = f"{split}_{metric}"
                    mean = curves[f"{key}_mean"].to_numpy(dtype=float)
                    x = curves.epoch.to_numpy()
                    curve_label = (f"{split_names[split]} ({group['label']})" if split == "train"
                                   else split_names[split])
                    axis.plot(x, mean, color=color, linestyle="-" if split == "train" else "--",
                              linewidth=2, label=curve_label)
                    if band == "sd":
                        delta = curves[f"{key}_sd"].to_numpy(dtype=float)
                        low, high = mean - delta, mean + delta
                    else:
                        low, high = (curves[f"{key}_{suffix}"].to_numpy(dtype=float) for suffix in ("ci_low", "ci_high"))
                    valid = (curves.n.to_numpy() >= 2) & np.isfinite(low) & np.isfinite(high)
                    if valid.any():
                        axis.fill_between(x, low, high, where=valid, color=color,
                                          alpha=.15 if split == "train" else .1)
            if reference is not None:
                values = reference["metrics"][metric]
                if values["low"] is not None:
                    axis.axhspan(values["low"], values["high"], color=PURPLE, alpha=.1, zorder=1)
                axis.axhline(values["mean"], color=PURPLE, linestyle="--", linewidth=2, zorder=3,
                             label=reference_label if reference_label else "_nolegend_")
            axis.set(title=title, xlabel="Эпоха обучения", ylabel=ylabel)
            add_panel_label(axis, 0 if metric == "loss" else 1, panel_positions)
            if xlim is None:
                axis.set_xlim(left=0)
            else:
                axis.set_xlim(*xlim)
            ylim = ylim_loss if metric == "loss" else ylim_accuracy
            if ylim is not None:
                axis.set_ylim(*ylim)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(True, linestyle=":", alpha=.6)
            axis.set_axisbelow(True)
            for spine in axis.spines.values():
                spine.set_color("#cccccc")
            if legend_loc is not None and (legend_on == "both" or
                    legend_on == ("left" if metric == "loss" else "right")):
                axis.legend(loc=legend_loc, ncol=legend_ncol, frameon=True)
        mixed = (any(group["protocol"].get("mixed_protocols", False) for group in prepared)
                 or bool(reference and reference.get("protocol", {}).get("mixed_protocols", False)))
        fig.tight_layout()
        for suffix in ("png", "svg"):
            fig.savefig(output / f"comparison_curves.{suffix}", dpi=600)
        curves_frame = pd.concat(frames, ignore_index=True)
        curves_frame.to_csv(output / "comparison_curves.csv", index=False)
        write_json(output / "comparison.json", dict(band=band, eval_split=eval_split, max_epochs=max_epochs, xlim=xlim,
                   ylim_loss=ylim_loss, ylim_accuracy=ylim_accuracy, legend_loc=legend_loc, legend_on=legend_on,
                   legend_ncol=legend_ncol,
                   panel_positions=panel_positions,
                   allow_mixed_protocols=allow_mixed_protocols,
                   mixed_protocols=mixed,
                   test_reference=reference,
                   series=[{key: value for key, value in group.items()
                            if key not in ("curves", "reference_curves")} for group in prepared],
                   notes=notes, interpretation="Полоса: выборочное SD по seed или 95%-й ДИ среднего по seed. "
                   "На каждой эпохе используются доступные запуски, без заполнения пропусков. "
                   "При одном seed полоса отсутствует. Test и val подписываются по фактическим данным."))
        if show:
            plt.show()
            plt.close(fig)
    for note in notes:
        print(note)
    return dict(figure=fig, curves=curves_frame, output=str(output), notes=notes,
                test_reference=reference, xlim=xlim, band=band)


def comparison_best_metrics(comparison):
    """Best aggregate train/test means in the plotted epoch range, plus fixed reference.

    Loss and accuracy optima are selected independently; ties use the earliest epoch.
    Bands describe the seeds at that epoch, rather than per-seed best values.
    """
    curves = comparison["curves"]
    xlim = comparison.get("xlim")
    if xlim is not None:
        curves = curves[curves.epoch.between(*xlim)]
    band = comparison.get("band", "sd")
    records, missing = [], []

    def record(series, split, metric, mean, low, high, epoch, n, criterion):
        spread = (high - low) / 2 if pd.notna(low) and pd.notna(high) else None
        records.append({"Серия": series, "Выборка": split, "Метрика": metric,
                        "Значение": mean, "Величина разброса (±)": spread,
                        "Нижняя граница": low, "Верхняя граница": high,
                        "Разброс": "SD" if band == "sd" else "95% ДИ",
                        "Эпоха": epoch, "Число запусков": n, "Критерий": criterion})

    for series, group in curves.groupby("series", sort=False):
        group = group.sort_values("epoch")
        for split, name in (("train", "Обучающая"), ("test", "Тестовая")):
            for metric, title in (("loss", "Функция потерь (Loss)"), ("accuracy", "Точность (Accuracy)")):
                key = f"{split}_{metric}"
                column = f"{key}_mean"
                if column not in group or not np.isfinite(group[column]).any():
                    missing.append(f"{series}: нет {split} {metric} по эпохам в выбранном диапазоне.")
                    continue
                valid = group[np.isfinite(group[column])]
                row = valid.loc[valid[column].idxmin() if metric == "loss" else valid[column].idxmax()]
                mean = float(row[column])
                if band == "sd":
                    delta = row[f"{key}_sd"]
                    low, high = (mean - delta, mean + delta) if pd.notna(delta) else (None, None)
                else:
                    low, high = row[f"{key}_ci_low"], row[f"{key}_ci_high"]
                record(series, name, title, mean, low, high, int(row.epoch), int(row.n),
                       "Минимум среднего" if metric == "loss" else "Максимум среднего")
    reference = comparison.get("test_reference")
    if reference is not None:
        for source, name in (("train_metrics", "Обучающая — ориентир"), ("metrics", "Тестовая — ориентир")):
            for metric, title in (("loss", "Функция потерь (Loss)"), ("accuracy", "Точность (Accuracy)")):
                values = reference.get(source, {}).get(metric)
                if values is None:
                    continue
                record(reference["series"], name, title, values["mean"],
                       values["low"], values["high"], reference["epoch"],
                       reference.get("n", len(reference["seeds"])), "Заданная эпоха ориентира")
    for message in missing:
        print(message)
    return pd.DataFrame(records, columns=["Серия", "Выборка", "Метрика", "Значение", "Величина разброса (±)",
                        "Нижняя граница", "Верхняя граница", "Разброс", "Эпоха", "Число запусков", "Критерий"])
