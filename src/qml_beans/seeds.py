"""Reproducible repeated-seed studies with validated collection and uncertainty reports."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pennylane as qml
from scipy.stats import t
import torch

from .data import prepare_cv_data, prepare_data
from .model import HybridClassifier, ModelConfig
from .training import fit, load_checkpoint, seed_everything, write_json


@dataclass(frozen=True)
class StageConfig:
    name: str = "hybrid"
    quantum: bool = True
    train_classical: bool = True
    train_quantum: bool = True
    epochs: int = 20
    batch_size: int = 64
    lr: float = 1e-3
    quantum_lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 7

    def __post_init__(self):
        if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]*", self.name):
            raise ValueError("Stage name must start with a letter and contain letters, digits, _ or -")
        if any(type(getattr(self, name)) is not bool for name in ("quantum", "train_classical", "train_quantum")):
            raise ValueError("Training switches must be booleans")
        if any(type(value) is not int or value < 1 for value in (self.epochs, self.batch_size)):
            raise ValueError("epochs and batch_size must be positive integers")
        if type(self.patience) is not int or self.patience < 0:
            raise ValueError("patience must be a nonnegative integer")
        if not all(np.isfinite(value) and value > 0 for value in (self.lr, self.quantum_lr)):
            raise ValueError("Learning rates must be positive and finite")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative and finite")


def parse_seeds(value):
    if isinstance(value, str):
        try:
            seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
        except ValueError as exc:
            raise ValueError("Seeds must be comma-separated integers, e.g. 11,22,33,44,55") from exc
    else:
        seeds = list(value)
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Provide at least one integer seed in 0..2**32-1")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds are not independent runs; remove duplicates")
    return seeds


def _json_value(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _versions():
    return dict(torch=str(torch.__version__), pennylane=str(qml.__version__), numpy=str(np.__version__))


def study_id(config):
    # Relocation of an identical source file does not change the scientific protocol.
    protocol = {key: value for key, value in config.items() if key != "source"}
    return sha256(json.dumps(protocol, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def make_study_config(source, model_config, stages, *, target="Class", deduplicate=True,
                      split_policy="fixed", split_seed=42, val_size=0.15, test_size=0.2, cv_folds=0):
    if split_policy not in ("fixed", "vary"):
        raise ValueError("split_policy must be fixed or vary")
    parse_seeds([split_seed])
    stages = [stage if isinstance(stage, StageConfig) else StageConfig(**stage) for stage in stages]
    if not stages or len({stage.name for stage in stages}) != len(stages):
        raise ValueError("Provide at least one stage with distinct stage names")
    config = model_config if isinstance(model_config, ModelConfig) else ModelConfig(**model_config)
    if type(cv_folds) is not int or cv_folds < 0 or cv_folds == 1:
        raise ValueError("cv_folds must be 0 (holdout) or an integer >= 2")
    if cv_folds:
        data = prepare_cv_data(source, target=target, seed=split_seed, deduplicate=deduplicate,
                               val_size=val_size, n_splits=cv_folds)[0]
        test_size = 1 / cv_folds
    else:
        data = prepare_data(source, target=target, seed=split_seed, deduplicate=deduplicate,
                            val_size=val_size, test_size=test_size)
    if config.n_features != data.x.shape[1] or config.n_classes != len(data.metadata["classes"]):
        raise ValueError("Model dimensions do not match the dataset")
    result = dict(format_version=1, source=str(Path(source).resolve()), target=target,
                            deduplicate=deduplicate, dataset_fingerprint=data.metadata["fingerprint"],
                            model_config=asdict(config), stages=[asdict(stage) for stage in stages],
                            split_policy=split_policy, split_seed=split_seed, val_size=val_size,
                            test_size=test_size, versions=_versions())
    if cv_folds:
        result["cv_folds"] = cv_folds
    return _json_value(result)


def export_study_config(config, path):
    write_json(path, config)
    return Path(path)


def _validate_config(config):
    if config.get("format_version") != 1:
        raise ValueError("Unsupported seed study format")
    rebuilt = make_study_config(config["source"], config["model_config"], config["stages"],
                                 target=config["target"], deduplicate=config["deduplicate"],
                                 split_policy=config["split_policy"], split_seed=config["split_seed"],
                                 val_size=config["val_size"], test_size=config["test_size"],
                                 cv_folds=config.get("cv_folds", 0))
    # Old analytic manifests predate the optional shot settings; retain their identity.
    for field in ("shots", "measurement_seed", "measurement_mode"):
        if field not in config["model_config"]:
            rebuilt["model_config"].pop(field, None)
    if study_id(rebuilt) != study_id(config):
        raise ValueError("Study source, configuration or library versions changed; create a new study")


def run_seed_study(config, seeds, output, *, skip_existing=False):
    """Append independently initialized seeds; stages within each seed transfer best weights."""
    seeds = parse_seeds(seeds)
    _validate_config(config)
    output = Path(output)
    identity = study_id(config)
    manifest = output / "study.json"
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8"))["study_id"] != identity:
            raise ValueError("Output contains a different dataset/model/protocol; choose another study directory")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Study output is nonempty and has no study.json; choose a new directory")
    # Preflight all requested directories before writing or training any seed.
    todo = []
    for seed in seeds:
        directory = output / f"seed_{seed}"
        if directory.exists():
            record = directory / "run.json"
            if not record.is_file():
                raise ValueError(f"Incomplete seed directory: {directory}. Choose a new output directory.")
            saved = json.loads(record.read_text(encoding="utf-8"))
            if saved["study_id"] != identity or saved["seed"] != seed:
                raise ValueError(f"Incompatible saved seed: {directory}")
            if not skip_existing:
                raise ValueError(f"Seed {seed} already exists; use skip_existing=True or choose other seeds")
            # A completion marker alone is insufficient if artifacts have since been removed.
            collect_seed_runs([directory])
            print(f"Seed {seed}: already complete, skipping", flush=True)
        else:
            todo.append(seed)
    output.mkdir(parents=True, exist_ok=True)
    write_json(manifest, dict(study_id=identity, config=config))
    export_study_config(config, output / "config.json")
    def partitions(split_seed):
        options = dict(target=config["target"], seed=split_seed, val_size=config["val_size"],
                        deduplicate=config["deduplicate"])
        if config.get("cv_folds"):
            return prepare_cv_data(config["source"], n_splits=config["cv_folds"], **options)
        return [prepare_data(config["source"], test_size=config["test_size"], **options)]

    fixed_data = partitions(config["split_seed"]) if config["split_policy"] == "fixed" else None
    for seed in todo:
        directory = output / f"seed_{seed}"
        directory.mkdir()
        split_seed = config["split_seed"] if fixed_data is not None else seed
        data_folds = fixed_data if fixed_data is not None else partitions(split_seed)
        completed_folds = []
        for fold, data in enumerate(data_folds):
            seed_everything(seed)
            model = HybridClassifier(ModelConfig(**config["model_config"]))
            fold_directory = directory / f"fold_{fold + 1:02d}" if config.get("cv_folds") else directory
            completed = []
            for index, stage_dict in enumerate(config["stages"]):
                stage = StageConfig(**stage_dict)
                model.set_mode(quantum=stage.quantum, train_classical=stage.train_classical,
                               train_quantum=stage.train_quantum)
                folder = f"{index + 1:02d}_{stage.name}"
                print(f"Seed {seed}, split seed {split_seed}, fold {fold + 1}/{len(data_folds)}, stage {stage.name}", flush=True)
                options = asdict(stage)
                for name in ("name", "quantum", "train_classical", "train_quantum"):
                    options.pop(name)
                fit(model, data, fold_directory / folder, seed=seed, **options,
                    export_test_predictions=bool(config.get("cv_folds")),
                    run_config=dict(source_data=config["source"], study_id=identity,
                                    split_seed=split_seed, stage=stage.name, fold=fold if config.get("cv_folds") else None))
                model, _ = load_checkpoint(fold_directory / folder / "best.pt")
                completed.append(dict(name=stage.name, folder=folder))
            completed_folds.append(dict(fold=fold, folder=f"fold_{fold + 1:02d}", stages=completed))
        # Written only after every stage succeeds. An interrupted run cannot enter a summary.
        saved = dict(format_version=1, study_id=identity, config=config, seed=seed, split_seed=split_seed)
        if config.get("cv_folds"):
            saved["folds"] = completed_folds
        else:
            saved["stages"] = completed
        write_json(directory / "run.json", saved)
    return output


def collect_seed_runs(paths):
    """Collect completed run directories or study roots, rejecting mixed protocols/duplicate seeds."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    records = {}
    for path in paths:
        path = Path(path).resolve()
        if path.name == "run.json" and path.is_file():
            candidates = [path]
        elif (path / "run.json").is_file():
            candidates = [path / "run.json"]
        elif (path / "study.json").is_file():
            candidates = sorted(path.glob("seed_*/run.json"))
        else:
            raise ValueError(f"Not a seed study or completed seed directory: {path}")
        for record in candidates:
            records[record.resolve()] = None
    if not records:
        raise ValueError("No completed seeds found")
    rows = []
    identities = set()
    seen_seeds = set()
    for record in records:
        saved = json.loads(record.read_text(encoding="utf-8"))
        config = saved["config"]
        identity = study_id(config)
        if saved.get("format_version") != 1 or saved["study_id"] != identity:
            raise ValueError(f"Invalid study identity in {record}")
        identities.add(identity)
        if len(identities) != 1:
            raise ValueError("Cannot pool different datasets, architectures, split policies or training protocols")
        seed = saved["seed"]
        parse_seeds([seed])
        if seed in seen_seeds:
            raise ValueError(f"Duplicate seed {seed} in different directories; refusing to inflate sample size")
        seen_seeds.add(seed)
        expected_split_seed = config["split_seed"] if config["split_policy"] == "fixed" else seed
        if config.get("cv_folds"):
            from .cross_validation import read_cv_run
            rows.extend(read_cv_run(config, saved, record, expected_split_seed))
            continue
        if saved["split_seed"] != expected_split_seed or len(saved["stages"]) != len(config["stages"]):
            raise ValueError(f"Invalid split/stage metadata in {record}")
        for index, (artifact, stage) in enumerate(zip(saved["stages"], config["stages"])):
            folder = f"{index + 1:02d}_{stage['name']}"
            if artifact != dict(name=stage["name"], folder=folder):
                raise ValueError(f"Invalid stage folder in {record}")
            directory = record.parent / folder
            metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            metadata = json.loads((directory / "data_metadata.json").read_text(encoding="utf-8"))
            if (metrics["run_config"].get("study_id") != identity or
                    metrics["run_config"]["seed"] != seed or
                    metrics["run_config"].get("split_seed") != expected_split_seed or
                    ModelConfig(**metrics["model_config"]) != ModelConfig(**config["model_config"]) or
                    metadata["fingerprint"] != config["dataset_fingerprint"] or
                    metadata["seed"] != expected_split_seed or
                    metadata["target"] != config["target"] or
                    metadata["deduplicate"] != config["deduplicate"] or
                    metadata["val_size"] != config["val_size"] or
                    metadata["test_size"] != config["test_size"] or
                    any(metrics["run_config"].get(name) != stage[name] for name in
                        ("epochs", "batch_size", "lr", "quantum_lr", "weight_decay", "patience")) or
                    any(metrics["mode"][name] != stage[name] for name in metrics["mode"])):
                raise ValueError(f"Incompatible artifacts in {directory}")
            history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
            if not (directory / "best.pt").is_file():
                raise ValueError(f"Missing best checkpoint: {directory}")
            rows.append(dict(seed=seed, split_seed=expected_split_seed, stage=stage["name"],
                             stage_index=index, best_epoch=metrics["best_epoch"],
                             n_train=metrics["splits"]["train"], n_val=metrics["splits"]["val"],
                             n_test=metrics["splits"]["test"],
                             val_accuracy=metrics["validation"]["accuracy"],
                             **{f"test_{name}": metrics["test"][name]
                                for name in ("accuracy", "balanced_accuracy", "macro_f1", "loss")},
                             output=str(directory), history=history))
    rows.sort(key=lambda row: (row["stage_index"], row["seed"]))
    return config, rows


def sample_statistics(values, *, confidence=0.95, allow_ci=True):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Statistics require a nonempty finite vector")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie between 0 and 1")
    n = len(values)
    result = dict(n=n, mean=float(values.mean()), sd=None, sem=None,
                  ci_low=None, ci_high=None, ci_half_width=None)
    if n > 1:
        sd = float(values.std(ddof=1))
        result.update(sd=sd)
        if allow_ci:
            sem = sd / np.sqrt(n)
            half = float(t.ppf((1 + confidence) / 2, n - 1) * sem)
            result.update(sem=float(sem), ci_low=result["mean"] - half,
                          ci_high=result["mean"] + half, ci_half_width=half)
    return result


def summarize_seed_runs(config, rows):
    summary = []
    for stage in config["stages"]:
        selected = [row for row in rows if row["stage"] == stage["name"]]
        for metric in ("test_accuracy", "test_balanced_accuracy", "test_macro_f1", "test_loss", "val_accuracy"):
            stats = sample_statistics([row[metric] for row in selected], allow_ci=config["split_policy"] == "fixed")
            summary.append(dict(stage=stage["name"], metric=metric, **stats))
    return summary


def _curves(rows, *, allow_ci=True):
    records = [dict(seed=row["seed"], **epoch) for row in rows for epoch in row["history"]]
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    records = []
    for epoch, selected in frame.groupby("epoch", sort=True):
        row = dict(epoch=int(epoch), n=int(len(selected)))
        for metric in ("train_loss", "val_loss", "train_accuracy", "val_accuracy"):
            stats = sample_statistics(selected[metric], allow_ci=allow_ci)
            for name in ("mean", "sd", "ci_low", "ci_high"):
                row[f"{metric}_{name}"] = stats[name]
        records.append(row)
    return pd.DataFrame(records)


def report_seed_study(paths, output, *, band="sd", show=True):
    """Export per-seed metrics, mean/SD/t-CI tables and plots with explicitly labelled bands."""
    config, rows = collect_seed_runs(paths)
    if band not in ("sd", "ci95"):
        raise ValueError("band must be sd or ci95")
    if band == "ci95" and config["split_policy"] != "fixed":
        raise ValueError("CI bands are offered only for a fixed split; use band='sd' for varying splits")
    if config.get("cv_folds"):
        from .cross_validation import report_cv_study
        return report_cv_study(config, rows, output, band=band, show=show)
    summary = summarize_seed_runs(config, rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{key: value for key, value in row.items() if key != "history"} for row in rows])
    frame.to_csv(output / "per_seed.csv", index=False)
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    interpretation = (
        "SD is sample standard deviation across seeds (ddof=1), not standard error. "
        "With a fixed split, the approximate 95% Student-t CI describes the mean over independent "
        "training seeds conditional on this dataset and split. It does not quantify uncertainty from "
        "sampling new test objects or prove a population accuracy. At n=1 SD/SEM/CI are undefined. "
        "With varying, overlapping splits only descriptive mean and SD are reported; SEM/CI are omitted. "
        "No best seed is selected. Early-stopped curves use available seeds only; n is shown per epoch."
    )
    write_json(output / "summary.json", dict(study_id=study_id(config), config=config, band=band,
                                            interpretation=interpretation, statistics=summary))
    statements = []
    for row in summary:
        if row["metric"] == "test_accuracy":
            text = f"{row['stage']}: test accuracy = {100 * row['mean']:.2f}%"
            if row["sd"] is None:
                text += "; n=1, uncertainty across seeds cannot be estimated"
            else:
                text += f" +/- {100 * row['sd']:.2f} percentage points (SD), n={row['n']}"
                if row["ci_low"] is not None:
                    text += f"; approximate 95% t-CI of seed mean [{100 * row['ci_low']:.2f}%, {100 * row['ci_high']:.2f}%]"
            statements.append(text)
    (output / "report.txt").write_text("\n".join(statements) + "\n\n" + interpretation + "\n", encoding="utf-8")
    _plot_seed_report(config, rows, summary, output, band, show)
    return dict(output=str(output), config=config, rows=frame, summary=pd.DataFrame(summary), statements=statements)


def _plot_seed_report(config, rows, summary, output, band, show, *, include_validation=False):
    from matplotlib.ticker import MaxNLocator
    if show:
        import matplotlib.pyplot as plt
        new_figure = plt.subplots
    else:
        # File export must not open a desktop window or require Tk/Qt.
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        def new_figure(nrows, ncols, **kwargs):
            figure = Figure(**kwargs)
            FigureCanvasAgg(figure)
            return figure, figure.subplots(nrows, ncols)

    error_name = "sd" if band == "sd" else "ci_half_width"
    band_label = ("Среднее ± выборочное стандартное отклонение" if band == "sd"
                  else "Среднее ± приближённый 95%-й ДИ (Стьюдент)")
    if len({row["seed"] for row in rows}) == 1:
        band_label = "Среднее, один запуск (разброс не определён)"
    stage_names = {"hybrid": "Гибридная модель", "classical": "Классическая модель",
                   "quantum_only": "Обучение квантовой части", "joint": "Совместное обучение"}
    stage_labels = [stage_names.get(stage["name"], f"Этап {index + 1}")
                    for index, stage in enumerate(config["stages"])]
    metric_labels = (("Вневыборочная точность (%)", "Вневыборочная макро-F1 (%)")
                     if config.get("cv_folds") else ("Точность на тестовой выборке (%)", "Макро-F1 на тестовой выборке (%)"))
    cv_note = f"; число блоков: {config['cv_folds']}" if config.get("cv_folds") else ""
    splits = [("test", "Тестовая выборка", "metrics_spread")]
    if include_validation:
        splits.append(("val", "Валидационная выборка", "metrics_spread_validation"))
    for prefix, split_label, filename in splits:
        fig, axes = new_figure(1, 2, figsize=(11, 4.5), constrained_layout=True)
        labels = (("Точность (%)", "Макро-F1 (%)") if include_validation else metric_labels)
        for axis, suffix, label in zip(axes, ("accuracy", "macro_f1"), labels):
            metric = f"{prefix}_{suffix}"
            for index, stage in enumerate(config["stages"]):
                selected = [row for row in rows if row["stage"] == stage["name"]]
                stats = next(row for row in summary if row["stage"] == stage["name"] and row["metric"] == metric)
                offsets = np.linspace(-0.10, 0.10, len(selected)) if len(selected) > 1 else [0]
                axis.scatter(index + np.asarray(offsets), [100 * row[metric] for row in selected],
                             color="tab:blue", alpha=0.65, s=30, label="Отдельные запуски" if index == 0 else None)
                error = stats[error_name]
                axis.errorbar(index, 100 * stats["mean"], yerr=None if error is None else 100 * error,
                              fmt="D", color="tab:orange", capsize=6, markersize=6, zorder=5,
                              label=band_label if index == 0 else None)
            axis.set_xticks(range(len(config["stages"])), stage_labels, rotation=15)
            axis.set(ylabel=label, title=label)
            axis.grid(axis="y", alpha=0.25)
            axis.legend(fontsize=8)
        split_note = f"\n{split_label}" if include_validation else ""
        fig.suptitle(f"Независимые запуски обучения: {len({row['seed'] for row in rows})}{cv_note}{split_note}")
        fig.savefig(output / f"{filename}.png", dpi=160)
        fig.savefig(output / f"{filename}.svg")
        if show:
            plt.show()
            plt.close(fig)
    for stage, stage_label in zip(config["stages"], stage_labels):
        selected = [row for row in rows if row["stage"] == stage["name"]]
        curves = _curves(selected, allow_ci=config["split_policy"] == "fixed")
        curves.to_csv(output / f"curves_{stage['name']}.csv", index=False)
        if curves.empty:
            continue
        fig, axes = new_figure(1, 2, figsize=(14, 3.8), constrained_layout=True)
        for axis, metrics, title in ((axes[0], ("train_loss", "val_loss"), "Кросс-энтропия"),
                                     (axes[1], ("train_accuracy", "val_accuracy"), "Точность")):
            for metric in metrics:
                mean = curves[f"{metric}_mean"].to_numpy(dtype=float)
                metric_label = "Обучающая выборка" if metric.startswith("train_") else "Валидационная выборка"
                line, = axis.plot(curves.epoch, mean, label=metric_label, marker="o", markersize=3)
                if band == "sd":
                    delta = curves[f"{metric}_sd"].to_numpy(dtype=float)
                    low, high = mean - delta, mean + delta
                else:
                    low = curves[f"{metric}_ci_low"].to_numpy(dtype=float)
                    high = curves[f"{metric}_ci_high"].to_numpy(dtype=float)
                axis.fill_between(curves.epoch, low, high, where=curves.n.to_numpy() >= 2,
                                  alpha=0.18, color=line.get_color())
            axis.set(title=title, xlabel="Эпоха")
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(alpha=0.2)
            axis.legend()
        # axes[2].step(curves.epoch, curves.n, where="mid", color="tab:green", marker="o", markersize=3)
        # axes[2].set(title="Available seeds (no forward-fill)", xlabel="Epoch", ylabel="n",
        #             ylim=(0, max(curves.n) + 1))
        # axes[2].grid(alpha=0.2)
        # axes[2].yaxis.set_major_locator(MaxNLocator(integer=True))
        # axes[2].xaxis.set_major_locator(MaxNLocator(integer=True))
        cv_note = "; в каждом запуске усреднены доступные блоки" if config.get("cv_folds") else ""
        fig.suptitle(f"{stage_label}: {band_label}{cv_note}")
        fig.savefig(output / f"learning_curves_{stage['name']}.png", dpi=160)
        fig.savefig(output / f"learning_curves_{stage['name']}.svg")
        if show:
            plt.show()
            plt.close(fig)
