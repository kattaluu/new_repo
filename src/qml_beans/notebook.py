"""Notebook helpers; the model, preprocessing and trainer are shared with the CLI."""

from datetime import datetime
import json
from pathlib import Path

import pandas as pd
import numpy as np
import torch

from .data import download_dry_bean, download_sonar, prepare_data
from .model import ACTIVATIONS, HybridClassifier, ModelConfig
from .training import fit, load_checkpoint, seed_everything


DATASETS = ("Dry Bean (UCI)", "Sonar (UCI)", "Свой CSV / ARFF")


def select_dataset(root, name, *, path="", target="Class", seed=42,
                   val_size=0.15, test_size=0.2, deduplicate=True):
    root = Path(root)
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    if name == DATASETS[0]:
        source = download_dry_bean(root / "data")
        target = "Class"
    elif name == "Sonar (UCI)":
        source = download_sonar(root / "data")
        target = "Class"
    elif name == DATASETS[-1]:
        if not str(path).strip():
            raise ValueError("Укажите путь к CSV/ARFF")
        source = Path(path).expanduser()
        if not source.is_absolute():
            source = root / source
    data = prepare_data(source, target=target, seed=seed, val_size=val_size,
                        test_size=test_size, deduplicate=deduplicate)
    return source, data


class LayerEditor:
    """One editable width and activation per hidden layer, including zero hidden layers."""

    def __init__(self, title, widths=(32,), activations=None):
        import ipywidgets as widgets

        self._widgets = widgets
        self.rows = []
        self.container = widgets.VBox()
        self.count = widgets.BoundedIntText(value=len(widths), min=0, max=12,
                                            description="Скрытых слоёв:", style={"description_width": "initial"})
        self.view = widgets.VBox([widgets.HTML(f"<b>{title}</b>"), self.count, self.container])
        for width, activation in zip(widths, activations or ("relu",) * len(widths)):
            self.rows.append(self._row(width, activation))
        self._refresh()
        self.count.observe(self._resize, names="value")

    def _row(self, width, activation):
        w = self._widgets
        return (w.BoundedIntText(value=width, min=1, max=4096, description="Нейронов:"),
                w.Dropdown(options=list(ACTIVATIONS), value=activation, description="Активация:"))

    def _refresh(self):
        self.container.children = tuple(self._widgets.HBox(row) for row in self.rows)

    def _resize(self, change):
        count = change["new"]
        self.rows = self.rows[:count]
        while len(self.rows) < count:
            self.rows.append(self._row(32, "relu"))
        self._refresh()

    def values(self):
        return tuple(row[0].value for row in self.rows), tuple(row[1].value for row in self.rows)


def parse_rotations(value):
    gates = tuple(part.strip().upper() for part in value.split(",") if part.strip())
    if not gates or any(gate not in ("RX", "RY", "RZ") for gate in gates):
        raise ValueError("Вращения: непустая последовательность RX, RY, RZ через запятую")
    return gates


def parse_edges(value):
    """Parse an ordered JSON list, never evaluate Python entered into a widget."""
    try:
        edges = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("Связи задаются JSON, например [[0,1],[1,2],[2,0]]") from exc
    if not isinstance(edges, list) or any(not isinstance(edge, list) for edge in edges):
        raise ValueError("Связи задаются списком пар, например [[0,1],[1,2]]")
    # ModelConfig validates wire indices and prohibits self loops.
    return tuple(tuple(edge) for edge in edges)


class Experiment:
    def __init__(self, root, source, data):
        self.root = Path(root)
        self.source = Path(source)
        self.data = data
        self.model = None
        self.runs = []

    def build(self, config, seed=42):
        if config.n_features != self.data.x.shape[1] or config.n_classes != len(self.data.metadata["classes"]):
            raise ValueError("Architecture dimensions do not match the selected dataset")
        seed_everything(seed)
        self.model = HybridClassifier(config)
        return self.model

    def restore(self, checkpoint_path):
        candidate, checkpoint = load_checkpoint(checkpoint_path)
        # Validate before replacing either model or preprocessing.
        restored_data = prepare_data(self.source, saved_metadata=checkpoint["data"])
        self.data, self.model = restored_data, candidate
        return self.model

    def train_stage(self, *, quantum=True, train_classical=True, train_quantum=True,
                    epochs=3, batch_size=64, lr=1e-3, quantum_lr=1e-3,
                    patience=7, seed=42, output=None):
        if self.model is None:
            raise ValueError("Сначала создайте модель или загрузите checkpoint")
        self.model.set_mode(quantum=quantum, train_classical=train_classical, train_quantum=train_quantum)
        output = Path(output) if output is not None else (
            self.root / "runs" / "notebook" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        metrics = fit(self.model, self.data, output, epochs=epochs, batch_size=batch_size,
                      lr=lr, quantum_lr=quantum_lr, patience=patience, seed=seed,
                      run_config={"source_data": str(self.source.resolve()), "interface": "notebook"})
        # The next stage always starts from the best validation weights of this stage.
        self.model, _ = load_checkpoint(output / "best.pt")
        result = dict(output=str(output), metrics=metrics)
        self.runs.append(result)
        return result

    def comparison(self):
        return pd.DataFrame([
            dict(run=index + 1, quantum=run["metrics"]["mode"]["quantum"],
                 train_classical=run["metrics"]["mode"]["train_classical"],
                 train_quantum=run["metrics"]["mode"]["train_quantum"],
                 best_epoch=run["metrics"]["best_epoch"],
                 val_accuracy=run["metrics"]["validation"]["accuracy"],
                 test_accuracy=run["metrics"]["test"]["accuracy"],
                 test_macro_f1=run["metrics"]["test"]["macro_f1"], output=run["output"])
            for index, run in enumerate(self.runs)
        ])


def plot_run(result):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    output = Path(result["output"])
    history = pd.read_json(output / "history.json")
    if not history.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), constrained_layout=True)
        for column in ("train_loss", "val_loss"):
            label = "Обучающая выборка" if column.startswith("train_") else "Валидационная выборка"
            axes[0].plot(history.epoch, history[column], label=label)
        for column in ("train_accuracy", "val_accuracy"):
            label = "Обучающая выборка" if column.startswith("train_") else "Валидационная выборка"
            axes[1].plot(history.epoch, history[column], label=label)
        for axis, title in zip(axes, ("Кросс-энтропия", "Точность")):
            axis.set(xlabel="Эпоха", title=title)
            axis.legend()
            axis.grid(alpha=0.2)
        fig.savefig(output / "learning_curves.png", dpi=150)
        plt.show()
    metrics = result["metrics"]
    fig, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ConfusionMatrixDisplay(confusion_matrix=np.asarray(metrics["test"]["confusion_matrix"]),
                           display_labels=metrics["classes"]).plot(ax=axis, cmap="Blues", colorbar=False,
                                                                    xticks_rotation=45)
    axis.set_title("Матрица ошибок на тестовой выборке\nЛучшая модель по валидационной выборке")
    axis.set_xlabel("Предсказанный класс")
    axis.set_ylabel("Истинный класс")
    fig.savefig(output / "confusion_matrix.png", dpi=150)
    plt.show()
