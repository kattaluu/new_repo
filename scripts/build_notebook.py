"""Regenerate the notebook source (does not execute or overwrite experiment artifacts)."""

import json
from pathlib import Path
from textwrap import dedent


cells = []


def cell(kind, text, tag):
    content = dict(cell_type=kind, id=tag, metadata={}, source=dedent(text).strip().splitlines(keepends=True))
    if kind == "code":
        content.update(execution_count=None, outputs=[])
    cells.append(content)




cell('markdown', r"""
# Конструктор гибридной квантовой модели

**8 кубитов · PyTorch + PennyLane · классические слои до и после схемы.**

Выполняйте ячейки сверху вниз (`Shift+Enter`). После ячейки с панелью измените
нужные значения, затем выполняйте следующую ячейку, которая применяет настройки.
Панель сама не запускает обучение. `Run All` выполнит пример: Dry Bean, 3 эпохи.

1. Выберите датасет и загрузите данные.
2. Настройте скрытые слои и квантовую схему, затем создайте модель.
3. Задайте режим обучения и запустите этап. Для следующего этапа измените только
   режим и повторите ячейку обучения: текущие лучшие веса сохраняются.
4. Для оценки разброса используйте секцию нескольких seed: каждый seed обучает новую модель.

Изменение архитектуры применяется при повторном выполнении **«Создать новую модель»**:
это создаёт новые случайные веса. Повторная загрузка данных также создаёт новый эксперимент.
Для продолжения существующего эксперимента используйте checkpoint.

Ноутбук использует `src/qml_beans` из этого проекта; сохраняйте структуру папок.
Предобработка обучается только на train. Val выбирает лучшие веса; test служит для
итогового отчёта каждого этапа. Не подбирайте архитектуру по результатам test.

### Число измерений (shots)

Поле **«Измерений (shots)»**: `0` — аналитическое среднее (`shots=None`);
положительное число, например `1024`, — столько повторений измерения на каждый объект.
Оценка Pauli Z равна $(N_0-N_1)/N$; для X/Y измерения выполняются в соответствующем базисе.
Все 8 кубитов измеряются совместно, один набор shots даёт 8 средних и сохраняет корреляции.
На векторе состояния shots моделируют статистику измерений идеальной схемы;
на матрице плотности — статистику измерений состояния с T1/T2.

Способ расчёта градиента выбирается в поле **«Режим измерений»**. Parameter-shift использует дополнительные сдвинутые схемы; два новых режима используют backprop через аналитическую симуляцию.
Параметры сохраняются в checkpoint, конфигурациях серий и метриках.
Повторная оценка может отличаться из-за статистического шума. «Seed измерений» фиксирует
поток случайных измерений; CPU-процессы получают разные подпотоки. Для воспроизводимости
полного запуска сохраняйте также число процессов, микробатч и порядок вычислений.


### Режим измерений и градиента

- **Выборочный forward + аналитический Jacobian** (`sampled_backprop`): на вход
  классической головы поступают настоящие совместные измерения восьми кубитов.
  Обратный проход использует Jacobian точного среднего, без parameter-shift.
  Это быстрый суррогатный градиент; он в общем случае смещён относительно градиента
  математического ожидания нелинейной конечной-shot функции потерь.
- **Аналитическое обучение, shots при оценке** (`analytic_train`): `model.train()`
  использует точные средние и backprop; `model.eval()` использует заданные shots.
  **Валидация и тест оба выборочные**, поэтому early stopping видит шумную оценку.
  `torch.no_grad()` сам по себе не переключает режим измерений.
- **Выборочные измерения + parameter-shift** (`parameter_shift`): прежний режим,
  выборочные средние и Jacobian через сдвинутые схемы. Оценка квантового Jacobian
  не гарантирует несмещённости градиента полной нелинейной конечной-shot функции потерь.

Все режимы поддерживают state vector и оба бэкенда матрицы плотности с T1/T2.
При 0 shots все режимы аналитические. Выбор режима, shots и seed входят в checkpoint
и протокол серии; разные режимы не объединяются в одну статистику.
""", 'intro')

cell('markdown', r"""
## Окружение

В терминале из корня проекта:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,notebook]"
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name qml-beans --display-name "Python (qml-beans)"
.\.venv\Scripts\python.exe -m jupyterlab notebooks/hybrid_qml_explorer.ipynb
```

В VS Code выберите ядро `.venv` / `Python (qml-beans)`. Если окружения ещё нет,
инструкция установки PyTorch есть в `README.md`. Нужен работающий Python kernel;
интерактивные панели не работают в статическом просмотрщике GitHub.
""", 'setup-help')

cell('code', r"""
from pathlib import Path
import os
import sys

# Работает при запуске Jupyter из корня проекта или папки notebooks.
ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents)
             if (p / "src" / "qml_beans" / "model.py").is_file()), None)
if ROOT is None:
    raise RuntimeError("Откройте ноутбук из папки проекта qml_beans")
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import json
import ipywidgets as w
from IPython.display import display
from IPython import get_ipython
get_ipython().run_line_magic("matplotlib", "inline")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import torch

from qml_beans.model import ACTIVATIONS, ModelConfig
from qml_beans.notebook import DATASETS, Experiment, LayerEditor, parse_edges, parse_rotations, plot_run, select_dataset
from qml_beans.data import class_distribution_table, load_table, prepare_cv_data, transform_features
from qml_beans.training import predict_logits
from qml_beans.seeds import StageConfig, export_study_config, make_study_config, parse_seeds, report_seed_study, run_seed_study

torch.set_num_threads(1)
print("Проект:", ROOT)
print("PyTorch:", torch.__version__, "| PennyLane:", qml.__version__)
from qml_beans.measurements import measurement_description
""", 'imports')

cell('markdown', r"""
## 1. Датасет

Dry Bean скачивается с [UCI](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
и кэшируется в `data/`. [UCI Sonar — Mines vs. Rocks](https://archive.ics.uci.edu/dataset/151/connectionist+bench+sonar+mines+vs+rocks)
также скачивается и кэшируется: 208 объектов, 60 признаков, метки M/R.
Для своего CSV/ARFF укажите файл и имя целевой колонки; остальные колонки должны
быть числовыми признаками. Путь может быть абсолютным или относительно корня проекта.
Поля «Файл» и «Метка» используются только для своего датасета.
Разбиение стратифицированное: сохраняются исходные доли классов, а не искусственные
50/50. Для Sonar это 111 M / 97 R. В таблице ниже видны количества, проценты и
отклонения после целочисленного округления; пересечение выборок запрещено.
""", 'dataset-help')

cell('code', r"""
dataset_choice = w.Dropdown(options=DATASETS, value=DATASETS[0], description="Датасет:", layout=w.Layout(width="400px"))
data_path = w.Text(value="", description="Файл:", placeholder="data/my_dataset.csv", layout=w.Layout(width="650px"))
target_column = w.Text(value="Class", description="Метка:")
split_seed = w.IntText(value=42, description="Seed:")
validation_fraction = w.BoundedFloatText(value=0.15, min=0.01, max=0.8, step=0.01, description="Val доля:")
test_fraction = w.BoundedFloatText(value=0.20, min=0.01, max=0.8, step=0.01, description="Test доля:")
deduplicate = w.Checkbox(value=True, description="Удалять точные дубликаты до разбиения", indent=False)
display(w.VBox([dataset_choice, data_path, target_column, w.HBox([split_seed, validation_fraction, test_fraction]), deduplicate]))
""", 'dataset-controls')

cell('code', r"""
# Загрузить данные / начать новый эксперимент с выбранными данными.
source, prepared = select_dataset(
    ROOT, dataset_choice.value, path=data_path.value, target=target_column.value,
    seed=split_seed.value, val_size=validation_fraction.value, test_size=test_fraction.value,
    deduplicate=deduplicate.value,
)
experiment = Experiment(ROOT, source, prepared)
print("Файл:", source)
print("Признаков:", prepared.x.shape[1], "| Классов:", len(prepared.metadata["classes"]))
print("Размеры:", {key: len(value) for key, value in prepared.metadata["indices"].items()})
print("Удалено дубликатов:", prepared.metadata["duplicates_dropped"])
display(pd.DataFrame({"class": prepared.metadata["classes"], "count": np.bincount(prepared.y)}))
display(class_distribution_table(prepared)[["split", "label", "count", "percent", "deviation_pp"]])
print("Проверки разбиения:", {key: value for key, value in prepared.metadata["split_audit"].items() if key != "rows"})
""", 'load-data')

cell('markdown', r"""
## 2. Классические блоки и квантовая схема

У каждого скрытого слоя — своё число нейронов и активация. Проекция в **8 признаков**
перед схемой и выходной слой в **число классов** добавляются автоматически: даже при
нуле скрытых слоёв остаются два полносвязных слоя. Выход модели — logits.

Кодирование: `R*(πz_i)`. Значение `z` определяется выбранной активацией проекции;
`tanh` ограничивает его диапазоном `[-1,1]`. Вращения в строке применяются **в указанном
порядке на каждом кубите**, затем выполняются связи. Один такой блок повторяется
«Глубина» раз. Для CNOT пара `[a,b]` означает control `a` → target `b`.

`custom` использует упорядоченный JSON-список пар индексов 0…7. `full` — все пары
`a<b`; `none` — без двухкубитных вентилей. Для RZ-кодирования включите начальные H,
чтобы входные данные влияли на относительные фазы. Повторный ввод кодирует те же
восемь признаков перед каждым следующим вариационным блоком.
""", 'architecture-help')

cell('code', r"""
pre_editor = LayerEditor("Скрытые слои ДО квантового блока", widths=(32,))
post_editor = LayerEditor("Скрытые слои ПОСЛЕ квантового блока", widths=(16,))
latent_activation = w.Dropdown(options=list(ACTIVATIONS), value="tanh", description="Активация → 8:", style={"description_width": "initial"})
quantum_depth = w.BoundedIntText(value=2, min=1, max=50, description="Глубина:")
embedding = w.Dropdown(options=["RX", "RY", "RZ"], value="RX", description="Кодирование:")
rotations = w.Text(value="RY,RX", description="Вращения:", placeholder="RY,RZ,RX")
topology = w.Dropdown(options=["ring", "linear", "brickwork", "brickwork3", "full", "none", "custom"], value="linear", description="Связи:")
entangler = w.Dropdown(options=["CNOT", "CZ"], value="CNOT", description="Вентиль:")
edges = w.Textarea(value="[[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]]", description="Custom:", layout=w.Layout(width="650px", height="70px"))
reupload = w.Checkbox(value=False, description="Повторный ввод данных в каждый блок", indent=False)
hadamard = w.Checkbox(value=False, description="Начальные H на всех кубитах", indent=False)
measurement = w.Dropdown(options=["X", "Y", "Z"], value="Z", description="Измерение:")
model_seed = w.IntText(value=42, description="Seed весов:")

def enable_h_for_rz(change):
    if change["new"] == "RZ":
        hadamard.value = True

embedding.observe(enable_h_for_rz, names="value")
display(w.VBox([
    pre_editor.view, latent_activation, post_editor.view,
    w.HTML("<b>Квантовая схема — 8 кубитов</b>"),
    w.HBox([quantum_depth, embedding]), rotations, w.HBox([topology, entangler]),
    edges, reupload, hadamard, w.HBox([measurement, model_seed]),
]))
measurement_shots = w.BoundedIntText(value=0, min=0, max=10000000, description="Измерений (shots):",
    style={"description_width": "initial"}, tooltip="0 — аналитическое среднее; например, 1024 — среднее по 1024 измерениям на объект")
measurement_seed = w.BoundedIntText(value=42, min=0, max=2**32-1, description="Seed измерений:",
    style={"description_width": "initial"})
display(w.HBox([measurement_shots, measurement_seed]))
measurement_mode = w.Dropdown(options=[
    ("Выборочные измерения + parameter-shift", "parameter_shift"),
    ("Выборочный forward + аналитический Jacobian", "sampled_backprop"),
    ("Аналитическое обучение, shots при оценке", "analytic_train"),
], value="parameter_shift", description="Режим измерений:",
   style={"description_width": "initial"}, layout=w.Layout(width="850px"))
display(measurement_mode)
display(w.HTML("<b>Выборочный forward + аналитический Jacobian:</b> реальные конечные измерения на входе "
               "головы, быстрый суррогатный backprop. <br><b>Аналитическое обучение:</b> точные средние "
               "при обучении; заданные shots при валидации и тестировании. <br>"
               "0 shots сохраняет аналитический режим во всех случаях. Настройки применяются при создании модели."))
""", 'architecture-controls')

cell('markdown', r"""
### Обучаемое кодирование

Угол RX/RY/RZ-кодирования: $\pi a_i z_i$, где $a_i$ — обучаемый масштаб кубита,
изначально равный 1. Добавляется 8 параметров квантовой части, общих для всех
повторных кодирований. Они обучаются и замораживаются вместе с квантовой частью.
Выключение опции сохраняет прежнее кодирование.
""", 'embedding-scale-help')

cell('code', r"""
trainable_embedding = w.Checkbox(value=True, description="Обучаемый масштаб кодирования",
    style={"description_width": "initial"}, layout=w.Layout(width="500px"))
display(trainable_embedding)
""", 'embedding-scale-control')

cell('code', r"""
# СОЗДАТЬ НОВУЮ МОДЕЛЬ. Повторное выполнение сбрасывает веса, но не удаляет сохранённые запуски.
pre_widths, pre_activations = pre_editor.values()
post_widths, post_activations = post_editor.values()
config = ModelConfig(
    n_features=experiment.data.x.shape[1], n_classes=len(experiment.data.metadata["classes"]),
    pre_layers=pre_widths, post_layers=post_widths,
    pre_activations=pre_activations, post_activations=post_activations,
    latent_activation=latent_activation.value, quantum_depth=quantum_depth.value,
    embedding=embedding.value, rotation_gates=parse_rotations(rotations.value),
    trainable_embedding=trainable_embedding.value,
    entanglement=topology.value, entangler=entangler.value,
    custom_edges=parse_edges(edges.value) if topology.value == "custom" else (),
    reupload=reupload.value, initial_hadamard=hadamard.value, measurement=measurement.value,
    shots=measurement_shots.value or None, measurement_seed=measurement_seed.value,
    measurement_mode=measurement_mode.value,
)
model = experiment.build(config, seed=model_seed.value)
print(model)
print("Полносвязных слоёв:", sum(isinstance(layer, torch.nn.Linear) for layer in model.modules()))
print("Всего параметров:", sum(p.numel() for p in model.parameters()))
print("Квантовых параметров:", model.quantum.weights.numel())
display(pd.Series(model.config_dict(), name="Созданная архитектура").to_frame())
print("Измерение:", measurement_description(config))
""", 'build-model')

cell('markdown', r"""
### Схема созданной модели

Рисунок показывает фактический QNode; после изменения настроек сначала создайте модель заново.
Он остаётся доступен и при выборе обхода квантового блока в режиме обучения.
""", 'circuit-help')

cell('code', r"""
with plt.rc_context({
    "font.family": "Times New Roman",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
}), torch.no_grad():
    figure, _ = qml.draw_mpl(
        experiment.model.quantum.circuit,
        decimals=None,
        fontsize=20,
        label_options={"fontsize": 20, "fontweight": "bold"},
    )(
        torch.zeros(8), experiment.model.quantum.weights.detach())
    for ax in figure.axes:
        for text in ax.texts:
            text.set_fontfamily("Times New Roman")
            if text.get_text() in {str(wire) for wire in range(8)}:
                text.set_text(r"$|0\rangle$")
            if text.get_text() in {"RX", "RY", "RZ", "H", "CNOT", "CZ", "X", "Y", "Z"}:
                text.set_fontweight("bold")
    display(figure)
    output_path = ROOT / "runs" / "quantum_circuit.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Сохранено: {output_path}")
    plt.close(figure)
""", 'draw-circuit')

cell('markdown', r"""
## 3. Обучение текущей модели

Три переключателя независимы. Если схема выключена, используется обход `8 → 8`;
её параметры не обновляются. Если заморожены все активные параметры, выполняется
оценка без оптимизации. Замороженная схема пропускает градиент к классическому encoder.

Пресеты меняют переключатели; обучение запускает **следующая ячейка**.
Для схемы «классика → квантовая → совместное» выберите первый пресет, запустите
этап, затем второй и снова запустите ту же ячейку, затем третий. Ячейку создания
модели между этапами не выполняйте. При включении схемы точность может упасть:
выход схемы отличается от тождественного обхода.

После каждого этапа в памяти остаются его лучшие веса по val loss. Каждый этап
использует новый Adam и новую папку результатов. Менять learning rate между этапами можно.
""", 'training-help')

cell('code', r"""
preset = w.Dropdown(options=["Вручную", "1. Классика без схемы", "2. Только квантовая", "3. Совместное", "Классика с фиксированной схемой"], value="3. Совместное", description="Режим:", layout=w.Layout(width="450px"))
use_quantum = w.Checkbox(value=True, description="Есть квантовый блок", indent=False)
train_classical = w.Checkbox(value=True, description="Классика обучается", indent=False)
train_quantum = w.Checkbox(value=True, description="Квантовая обучается", indent=False)
epochs = w.BoundedIntText(value=100, min=1, max=10000, description="Эпох:")
batch_size = w.BoundedIntText(value=64, min=1, max=2048, description="Batch size:")
learning_rate = w.BoundedFloatText(value=0.001, min=1e-8, max=1, step=0.0001, description="LR классики:")
quantum_learning_rate = w.BoundedFloatText(value=0.001, min=1e-8, max=1, step=0.0001, description="LR схемы:")
patience = w.BoundedIntText(value=100, min=0, max=10000, description="Patience:")
training_seed = w.IntText(value=42, description="Seed этапа:")

def apply_preset(change):
    choices = {
        "1. Классика без схемы": (False, True, False),
        "2. Только квантовая": (True, False, True),
        "3. Совместное": (True, True, True),
        "Классика с фиксированной схемой": (True, True, False),
    }
    if change["new"] in choices:
        use_quantum.value, train_classical.value, train_quantum.value = choices[change["new"]]

preset.observe(apply_preset, names="value")
display(w.VBox([preset, w.HBox([use_quantum, train_classical, train_quantum]),
                w.HBox([epochs, batch_size, patience]),
                w.HBox([learning_rate, quantum_learning_rate, training_seed])]))
""", 'training-controls')

cell('code', r"""
# ЗАПУСТИТЬ ЭТАП. Повторяйте эту ячейку для продолжения с текущими лучшими весами.
result = experiment.train_stage(
    quantum=use_quantum.value, train_classical=train_classical.value, train_quantum=train_quantum.value,
    epochs=epochs.value, batch_size=batch_size.value, lr=learning_rate.value,
    quantum_lr=quantum_learning_rate.value, patience=patience.value, seed=training_seed.value,
)
print("Результаты:", result["output"])
plot_run(result)
display(experiment.comparison())
report = result["metrics"]["test"]["classification_report"]
display(pd.DataFrame({key: value for key, value in report.items() if isinstance(value, dict)}).T)
""", 'train-stage')

cell('markdown', r"""
## 4. Несколько seed: среднее, разброс и интервал среднего

### Необязательная Stratified k-fold кросс-валидация

Включите флажок k-fold ниже и задайте `k` (например 5); можно указать **один seed**
или несколько. Без флажка остаётся прежнее holdout-разбиение.
Одиночное обучение секции 3 остаётся holdout; k-fold запускается этой секцией.

Каждый внешний фолд служит test ровно один раз. Из остальных данных стратифицированно
выделяется validation (поле «CV val» — доля **от оставшейся внешней train-части**).
Test доля из секции 1 в этом режиме не используется: её задаёт `k`.
Для каждого seed × fold создаются новые веса и свой scaler/медианы, обученные только
на train. Этапы переносят веса только **внутри одного фолда**, без переноса между фолдами.
Перед запуском выводится аудит M/R для всех фолдов выбранного разбиения.

Сводка k-fold использует **OOF**: у каждого объекта ровно одно тестовое предсказание
на seed и этап. Accuracy/F1 считаются по всем OOF-предсказаниям, а не простым средним
F1 фолдов. Отдельно сохраняются метрики фолдов и их описательный SD.
Фолды зависимы из-за пересечения train: CI по `seed × fold` не вычисляется.
Для разброса/CI итоговой OOF-метрики `n` равно **числу seed**, а не `seed × k`.
При одном seed виден результат CV и разброс фолдов, но SD/CI по seed не определены.
Графики обучения сначала усредняют доступные фолды внутри seed, затем seed;
число доступных фолдов сохраняется в `available_folds_per_epoch.csv`.

Это внешняя оценка фиксированного протокола с внутренним validation для early stopping,
а не nested-поиск гиперпараметров. Не выбирайте архитектуру по внешним test/OOF-метрикам.
CV-серия сохраняет отдельные checkpoint фолдов и оставляет одиночную модель секции 3
для предсказаний. Для применения весов конкретного фолда загрузите его checkpoint явно.

Секция использует **созданную архитектуру** и **загруженный датасет**. Для каждого seed
создаётся новая модель; текущие обученные веса не используются. Внутри одного seed
можно обучить один этап с текущими переключателями или весь протокол
«классика → только квантовая → совместное». Между этапами переносятся лучшие веса по val loss.

По умолчанию seed меняет инициализацию и порядок батчей; train/val/test фиксированы
seed из секции данных. Это измеряет случайность обучения на одном test. Режим `vary`
дополнительно меняет разбиение (и предобработку train) для каждого seed.

- **SD** — стандартное отклонение результатов между seed (`ddof=1`), не погрешность среднего.
- **SEM** — `SD / sqrt(n)`; при фиксированном разбиении приблизительный **95% t-CI**
  среднего: `mean ± t(0.975, n-1) × SEM`. `n` — число независимых запусков, не объектов test.
- При **одном seed** SD/SEM/CI не определены, а не равны нулю.
- T-интервал приблизителен и предполагает независимость запусков и применимость
  нормальной модели для метрик; при малом числе seed его следует трактовать осторожно.
- При `vary` выводятся только описательные mean/SD: перекрывающиеся разбиения
  не дают здесь формальной оценки погрешности обобщения.

Даже CI при фиксированном test относится к среднему по случайности обучения на
данном датасете и разбиении. Он **не учитывает выбор новых тестовых объектов** и не
доказывает «истинную точность ± погрешность». Для заявления указывайте метрику,
SD или CI, число seed и способ разбиения. Желательно заранее выбрать 5–10 или больше
seed и протокол; не отбирайте лучший seed и не подбирайте настройки по test.

Графики показывают отдельные seed, среднее с выбранными усами и кривые с полосами.
При early stopping эпохи усредняются по доступным запускам: число `n` показывается
отдельно, завершённые кривые не продолжаются искусственно.

Запуски внутри пакета выполняются последовательно. Можно дополнять ту же папку
новыми seed в другой сессии или собирать совместимые отдельные папки ниже.
Повторные seed не считаются новыми наблюдениями; несовместимые протоколы не смешиваются.
""", 'seeds-help')

cell('code', r"""
study_seeds = w.Text(value="11,22,33,44,55", description="Seeds:", layout=w.Layout(width="550px"))
study_cv = w.Checkbox(value=False, description="Stratified k-fold кросс-валидация", indent=False)
study_k = w.BoundedIntText(value=5, min=2, max=20, description="k фолдов:")
study_cv_val = w.BoundedFloatText(value=0.15, min=0.01, max=0.8, step=0.01, description="CV val:")
study_policy = w.Dropdown(options=[("Фиксированное разбиение", "fixed"), ("Разбиение меняется с seed", "vary")], value="fixed", description="Разбиение:", layout=w.Layout(width="450px"))
study_protocol = w.Dropdown(options=[("Один этап: текущие переключатели", "single"), ("Классика → квантовая → совместное", "staged")], value="single", description="Протокол:", layout=w.Layout(width="450px"))
study_classical_epochs = w.BoundedIntText(value=20, min=1, max=10000, description="Классика эпох:", style={"description_width": "initial"})
study_quantum_epochs = w.BoundedIntText(value=5, min=1, max=10000, description="Квантовая эпох:", style={"description_width": "initial"})
study_joint_epochs = w.BoundedIntText(value=20, min=1, max=10000, description="Совместно эпох:", style={"description_width": "initial"})
study_joint_lr = w.BoundedFloatText(value=0.0003, min=1e-8, max=1, step=0.0001, description="Joint LR:")
study_output = w.Text(value="runs/seed_study_sonar_kfold_", description="Папка серии:", style={"description_width": "initial"}, layout=w.Layout(width="750px"))
study_band = w.Dropdown(options=[("Среднее ± SD", "sd"), ("Среднее ± 95% t-CI", "ci95")], value="sd", description="Полоса:")
study_skip = w.Checkbox(value=True, description="Пропускать уже завершённые seed", indent=False)
study_run = w.Checkbox(value=False, description="Обучить указанные seed при выполнении следующей ячейки", indent=False)

def update_study_policy(change):
    if change["new"] == "vary":
        study_band.value = "sd"
    study_band.disabled = change["new"] == "vary"

study_policy.observe(update_study_policy, names="value")
display(w.VBox([study_seeds, study_cv, w.HBox([study_k, study_cv_val]), study_policy, study_protocol,
                w.HBox([study_classical_epochs, study_quantum_epochs, study_joint_epochs]),
                study_joint_lr, study_output, study_band, study_skip, study_run]))
""", 'seeds-controls')

cell('code', r"""
# Конфигурация всегда экспортируется; обучение выполняется только при включённом флажке.
from datetime import datetime
selected_seeds = parse_seeds(study_seeds.value)
common = dict(batch_size=batch_size.value, lr=learning_rate.value,
              quantum_lr=quantum_learning_rate.value, patience=patience.value)
if study_protocol.value == "staged":
    stages = [
        StageConfig(name="classical", quantum=False, train_quantum=False,
                    epochs=study_classical_epochs.value, **common),
        StageConfig(name="quantum_only", train_classical=False,
                    epochs=study_quantum_epochs.value, **common),
        StageConfig(name="joint", epochs=study_joint_epochs.value,
                    **dict(common, lr=study_joint_lr.value)),
    ]
else:
    stages = [StageConfig(name="hybrid" if use_quantum.value else "classical",
                          quantum=use_quantum.value, train_classical=train_classical.value,
                          train_quantum=train_quantum.value, epochs=epochs.value, **common)]
metadata = experiment.data.metadata
study_config = make_study_config(
    experiment.source, experiment.model.config, stages,
    target=metadata["target"], deduplicate=metadata["deduplicate"],
    split_policy=study_policy.value, split_seed=metadata["seed"],
    val_size=study_cv_val.value if study_cv.value else metadata.get("val_size", len(metadata["indices"]["val"]) / len(experiment.data.y)),
    test_size=metadata.get("test_size", len(metadata["indices"]["test"]) / len(experiment.data.y)),
    cv_folds=study_k.value if study_cv.value else 0,
)
if study_cv.value:
    preview_seed = metadata["seed"] if study_policy.value == "fixed" else selected_seeds[0]
    preview_folds = prepare_cv_data(experiment.source, target=metadata["target"], n_splits=study_k.value,
                                    seed=preview_seed, val_size=study_cv_val.value,
                                    deduplicate=metadata["deduplicate"])
    preview_classes = pd.concat([class_distribution_table(data).assign(fold=index + 1)
                                 for index, data in enumerate(preview_folds)], ignore_index=True)
    display(preview_classes[["fold", "split", "label", "count", "percent", "deviation_pp"]])
    print("Стратификация в пределах округления одного объекта:",
          all(all(data.metadata["split_audit"]["within_one_object"].values()) for data in preview_folds))
config_file = ROOT / "runs" / "notebook_configs" / ("study_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
export_study_config(study_config, config_file)
study_folder = Path(study_output.value).expanduser()
if not study_folder.is_absolute():
    study_folder = ROOT / study_folder
print("Конфигурация:", config_file)
print("Протокол:", [stage.name for stage in stages], "| Seeds:", selected_seeds)
if study_run.value:
    run_seed_study(study_config, selected_seeds, study_folder, skip_existing=study_skip.value)
    seed_report = report_seed_study([study_folder], study_folder / "report", band=study_band.value)
    display(seed_report["rows"])
    display(seed_report["summary"])
    if "fold_summary" in seed_report:
        display(seed_report["fold_summary"])
    print("\n".join(seed_report["statements"]))
else:
    print("Обучение серии выключено. Для запуска включите флажок выше и повторите эту ячейку.")
    print("Отдельный запуск одного seed из терминала (замените seed и папку при необходимости):")
    print(f'"{sys.executable}" scripts/run_seeds.py run --config "{config_file}" --seeds {selected_seeds[0]} --output "{study_folder}"')
""", 'run-seeds')

cell('markdown', r"""
### Собрать отдельные запуски без повторного обучения

Укажите по одной папке серии или завершённого seed на строку. Можно собрать seed,
обученные в отдельных папках с одной экспортированной конфигурацией. Таблицы и
графики пересоздаются по сохранённым результатам; обучение не запускается.
Пустое поле пропускает сборку. Для другой величины полосы используйте переключатель выше.
""", 'collect-help')

cell('code', r"""
collect_paths = w.Textarea(value="", description="Папки:", placeholder="runs/seed_study\nruns/seed_study_other", layout=w.Layout(width="750px", height="90px"))
collect_output = w.Text(value="runs/seed_collected", description="Отчёт:", layout=w.Layout(width="750px"))
display(w.VBox([collect_paths, collect_output]))
""", 'collect-controls')

cell('code', r"""
if collect_paths.value.strip():
    inputs = [Path(line.strip()).expanduser() for line in collect_paths.value.splitlines() if line.strip()]
    inputs = [path if path.is_absolute() else ROOT / path for path in inputs]
    destination = Path(collect_output.value).expanduser()
    if not destination.is_absolute():
        destination = ROOT / destination
    collected = report_seed_study(inputs, destination, band=study_band.value)
    display(collected["rows"])
    display(collected["summary"])
    if "fold_summary" in collected:
        display(collected["fold_summary"])
    print("\n".join(collected["statements"]))
else:
    print("Папки не указаны — сборка пропущена.")
""", 'collect-seeds')

cell('markdown', r"""
## 5. Продолжить с сохранённых весов

Укажите `best.pt` или `last.pt` и выполните следующую ячейку. Датасет выше должен
быть тем же, что при сохранении checkpoint: проверяются содержимое и порядок строк.
Архитектура, классы, предобработка и разбиение восстановятся из checkpoint;
значения панелей создания модели при этом не применяются.
После загрузки выбирайте режим в секции 3 и повторяйте ячейку запуска этапа.
Пустой путь оставляет текущую модель без изменений.
""", 'checkpoint-help')

cell('code', r"""
checkpoint_path = w.Text(value="", description="Checkpoint:", placeholder="runs/demo/classical/best.pt", layout=w.Layout(width="800px"))
display(checkpoint_path)
""", 'checkpoint-control')

cell('code', r"""
if checkpoint_path.value.strip():
    checkpoint_file = Path(checkpoint_path.value).expanduser()
    if not checkpoint_file.is_absolute():
        checkpoint_file = ROOT / checkpoint_file
    experiment.restore(checkpoint_file)
    measurement_mode.value = experiment.model.config.measurement_mode
    measurement_shots.value = experiment.model.config.shots or 0
    measurement_seed.value = experiment.model.config.measurement_seed
    trainable_embedding.value = experiment.model.config.trainable_embedding
    mode = experiment.model.mode_dict()
    preset.value = "Вручную"
    use_quantum.value = mode["quantum"]
    train_classical.value = mode["train_classical"]
    train_quantum.value = mode["train_quantum"]
    print("Загружено:", checkpoint_file)
    print(experiment.model)
    display(pd.Series(experiment.model.config_dict(), name="Восстановленная архитектура").to_frame())
else:
    print("Путь не указан — используется текущая модель.")
""", 'restore-checkpoint')

cell('markdown', r"""
## 6. Конфигурация и предсказания

Полная конфигурация сохраняется в каждом `best.pt` и `metrics.json`.
В папке этапа также находятся `last.pt`, история, матрица ошибок и графики PNG.
Веса выбранной текущей модели можно применять к новым объектам: код ниже выполняется
только при указанном пути. Нужны те же признаки, порядок колонок может отличаться,
целевая колонка необязательна. Для смены датасета начните новый эксперимент в секции 1.

Для настройки из кода доступен `ModelConfig(...)`: например,
`pre_layers=(64,32), pre_activations=("gelu","tanh"), post_layers=(), post_activations=()`.
Связи задаются `entanglement="custom", custom_edges=((0,1),(0,3),(3,7))`.
Настройки квантового блока описывают повторяемый вариационный блок; для произвольной
последовательности других вентилей редактируйте `QuantumBlock` в `src/qml_beans/model.py`.
""", 'exports-help')

cell('code', r"""
print(json.dumps(experiment.model.config_dict(), ensure_ascii=False, indent=2))
prediction_path = w.Text(value="", description="Новые данные:", placeholder="data/new_beans.csv", style={"description_width": "initial"}, layout=w.Layout(width="800px"))
display(prediction_path)
""", 'prediction-control')

cell('code', r"""
if prediction_path.value.strip():
    new_file = Path(prediction_path.value).expanduser()
    if not new_file.is_absolute():
        new_file = ROOT / new_file
    values = transform_features(load_table(new_file), experiment.data.metadata["preprocessing"])
    probabilities = predict_logits(experiment.model, values, batch_size.value).softmax(-1).numpy()
    classes = experiment.data.metadata["classes"]
    predictions = pd.DataFrame({"prediction": [classes[index] for index in probabilities.argmax(-1)]})
    for index, name in enumerate(classes):
        predictions[f"probability_{name}"] = probabilities[:, index]
    display(predictions.head(20))
    # При необходимости: predictions.to_csv(ROOT / "runs" / "predictions.csv", index=False)
else:
    print("Для предсказаний укажите файл в поле выше и повторите эту ячейку.")
""", 'predict-new')

cell('markdown', r"""
Данные Dry Bean: *Dry Bean [Dataset]* (2020), UCI,
[DOI 10.24432/C50S4B](https://doi.org/10.24432/C50S4B), CC BY 4.0.
Данные Sonar: Sejnowski, T. & Gorman, R. (1988), *Connectionist Bench (Sonar, Mines vs. Rocks)*,
[DOI 10.24432/C5T01Q](https://doi.org/10.24432/C5T01Q), CC BY 4.0.
Формула t-интервала среднего: [NIST](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm).
Стратифицированная CV: [scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html).
Документация: [PennyLane + PyTorch](https://docs.pennylane.ai/en/stable/introduction/interfaces/torch.html),
[Jupyter Widgets](https://ipywidgets.readthedocs.io/en/stable/).
""", 'references')


if __name__ == "__main__":
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells,
                    metadata=dict(kernelspec=dict(display_name="Python (qml-beans)",
                                                  language="python", name="qml-beans"),
                                  language_info=dict(name="python", version="3.12")))
    destination = Path(__file__).resolve().parents[1] / "notebooks" / "hybrid_qml_explorer.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(destination)
