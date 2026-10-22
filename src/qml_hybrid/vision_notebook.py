"""Circuit and training plots for code-only notebooks."""

from pathlib import Path
import numpy as np
import pandas as pd
import torch

def draw_vision_circuit(model, output=None, *, show_noise=False, fontsize=20):
    """Draw the configured vision circuit without executing it or changing model/RNG state."""
    import matplotlib.pyplot as plt
    import pennylane as qml
    from .noisy import circuit_schedule

    config, noise = model.config, model.noise
    rotations = {"RX": qml.RX, "RY": qml.RY, "RZ": qml.RZ}

    def circuit(inputs, weights):
        for gate, info, duration in circuit_schedule(config, noise):
            if gate in ("CNOT", "CZ", "iSWAP", "CNOT_parallel", "CZ_parallel", "iSWAP_parallel"):
                edges = info if gate.endswith("_parallel") else (info,)
                for edge in edges:
                    {"CNOT": qml.CNOT, "CZ": qml.CZ, "iSWAP": qml.ISWAP}[gate.removesuffix("_parallel")](wires=edge)
            elif gate != "idle":
                for wire in range(8):
                    if gate == "H":
                        qml.Hadamard(wires=wire)
                    elif gate == "embed":
                        rotations[config.embedding](torch.pi * inputs[wire], wires=wire)
                    else:
                        layer, gate_index = info
                        rotations[gate](weights[layer, wire, gate_index], wires=wire)
            if show_noise and noise.enabled and duration:
                for wire in range(8):
                    qml.ThermalRelaxationError(0.0, noise.t1_us[wire], noise.t2_us[wire], duration, wires=wire)
        observable = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}[config.measurement]
        return tuple(qml.expval(observable(wire)) for wire in range(8))

    with plt.rc_context({
        "font.family": "Times New Roman", "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
    }), torch.no_grad():
        figure, axes = qml.draw_mpl(circuit, decimals=None, fontsize=fontsize,
                                    label_options={"fontsize": fontsize, "fontweight": "bold"})(
                                        torch.zeros(8), model.quantum.weights.detach())
        for axis in figure.axes:
            for text in axis.texts:
                text.set_fontfamily("Times New Roman")
                if text.get_text() in {str(wire) for wire in range(8)}:
                    text.set_text(r"$|0\rangle$")
                if text.get_text() in {"RX", "RY", "RZ", "H", "CNOT", "CZ", "iSWAP", "ISWAP", "X", "Y", "Z"}:
                    text.set_fontweight("bold")
        if output is not None:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output, dpi=300, bbox_inches="tight")
    return figure, axes


def plot_run(result):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    output = Path(result["output"])
    history = pd.read_json(output / "history.json")
    if not history.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), constrained_layout=True)
        split_labels = dict(train="Обучающая выборка", val="Валидационная выборка", test="Тестовая выборка")
        for column in ("train_loss", "val_loss", "test_loss"):
            if column not in history:
                continue
            label = split_labels[column.split("_")[0]]
            axes[0].plot(history.epoch, history[column], label=label)
        for column in ("train_accuracy", "val_accuracy", "test_accuracy"):
            if column not in history:
                continue
            label = split_labels[column.split("_")[0]]
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
    selection_label = ("Лучшая модель по валидационной выборке" if metrics.get("validation") is not None
                       else "Модель последней эпохи")
    axis.set_title("Матрица ошибок на тестовой выборке\n" + selection_label)
    axis.set_xlabel("Предсказанный класс")
    axis.set_ylabel("Истинный класс")
    fig.savefig(output / "confusion_matrix.png", dpi=150)
    plt.show()
