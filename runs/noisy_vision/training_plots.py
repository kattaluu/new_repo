import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

MODE = 2
folder = Path(__file__).resolve().parent

# 1. Загрузка данных из CSV-файлов (MODE = 1 - MNIST и f-MNIST, != 1 CIFAR-10)
if MODE == 1:
    df_1 = pd.read_csv(folder / "mnist_no_noise" / "curves_hybrid.csv")
    df_2 = pd.read_csv(folder / "fmnist_no_noise" / "curves_hybrid.csv")
    ds_label = {0: "MNIST", 1: "f-MNIST"}
    colors = {"loss": {0: "#1f77b4", 1: "#ff7f0e"},
              "accuracy": {0: "#2ca02c", 1: "#d62728"}}
else:
    df_1 = pd.read_csv(folder / "cifar-10_classic" / "curves_classical.csv")
    df_2 = pd.read_csv(folder / "cifar-10_hybrid" / "curves_hybrid.csv")
    ds_label = {0: "CIFAR-10 classic", 1: "CIFAR-10 hybrid"}
    colors = {"loss": {0: "#008080", 1: "#d32f2f"},
              "accuracy": {0: "#e6ab02", 1: "#7570b3"}}

# Настройка стиля и шрифтов для поддержки кириллицы
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"font.size": 14, "figure.titlesize": 14})

# Создаем общую фигуру с двумя подграфиками (1 строка, 2 колонки)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))


# --- ГРАФИК 1: ФУНКЦИЯ ПОТЕРЬ (LOSS) ---
for i, df in enumerate([df_1, df_2]):
    ax1.plot(
        df["epoch"],
        df["train_loss_mean"],
        label=f"Обучающая выборка ({ds_label[i]})",
        color=colors["loss"][i],
        linewidth=2,
    )
    ax1.plot(
        df["epoch"],
        df["val_loss_mean"],
        label=f"Валидационная выборка ({ds_label[i]})",
        color=colors["loss"][i],
        linestyle="--",
        linewidth=2,
    )
    ax1.fill_between(df["epoch"], 
                    df["train_loss_mean"] - df["train_loss_sd"], 
                    df["train_loss_mean"] + df["train_loss_sd"], 
                    color=colors["loss"][i], alpha=0.15)
    ax1.fill_between(df["epoch"], 
                    df["val_loss_mean"] - df["val_loss_sd"], 
                    df["val_loss_mean"] + df["val_loss_sd"], 
                    color=colors["loss"][i], alpha=0.1)

ax1.set_title("Функция потерь (Loss)")
ax1.set_xlabel("Эпоха обучения", fontsize=14)
ax1.set_ylabel("Значение кросс-энтропии", fontsize=14)
ax1.legend(frameon=True)
ax1.grid(True, linestyle=":", alpha=0.6)

# --- ГРАФИК 2: ТОЧНОСТЬ (ACCURACY) ---
for i, df in enumerate([df_1, df_2]):
    ax2.plot(
        df["epoch"],
        df["train_accuracy_mean"],
        label=f"Обучающая выборка ({ds_label[i]})",
        color=colors["accuracy"][i],
        linewidth=2,
    )
    ax2.plot(
        df["epoch"],
        df["val_accuracy_mean"],
        label=f"Валидационная выборка ({ds_label[i]})",
        color=colors["accuracy"][i],
        linestyle="--",
        linewidth=2,
    )
    ax2.fill_between(df["epoch"], 
                    df["train_accuracy_mean"] - df["train_accuracy_sd"], 
                    df["train_accuracy_mean"] + df["train_accuracy_sd"], 
                    color=colors["accuracy"][i], alpha=0.15)
    ax2.fill_between(df["epoch"], 
                    df["val_accuracy_mean"] - df["val_accuracy_sd"], 
                    df["val_accuracy_mean"] + df["val_accuracy_sd"], 
                    color=colors["accuracy"][i], alpha=0.1)

ax2.set_title("Точность (Accuracy)")
ax2.set_xlabel("Эпоха обучения")
ax2.set_ylabel("Значение точности")
ax2.legend(frameon=True)
ax2.grid(True, linestyle=":", alpha=0.6)

# Оптимизация расположения элементов
plt.tight_layout()

# Если нужно сохранить график в файл, раскомментируйте строку ниже:
fig.savefig(folder / f"learning_curves_{MODE}.png", dpi=600)