"""Build the separate Russian-language image/noise/CPU notebook."""

import json
from pathlib import Path
from textwrap import dedent

cells = []


def cell(kind, source, identifier):
    item = dict(cell_type=kind, id=identifier, metadata={}, source=dedent(source).strip() + "\n")
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    cells.append(item)





cell('markdown', r"""
# Изображения: гибридная модель с T1/T2 и параллельным обучением на CPU

Отдельный эксперимент: **MNIST, Fashion MNIST, CIFAR-10**, 8 кубитов,
полносвязные слои до и после квантовой схемы. Исходный ноутбук для табличных данных сохранён.
Выполняйте ячейки по порядку; повторное обучение запускает только ячейка «Обучение».

Бэкенды:

- `torch_density`: специализированная матрица плотности 256×256, автодифференцирование PyTorch;
- `pennylane_mixed`: эталонный `default.mixed`, `ThermalRelaxationError`, `backprop`;
- `pennylane_statevector`: идеальный `default.qubit` без T1/T2, для сравнения с шумовыми моделями.
- `aer_statevector`: Qiskit Aer, без T1/T2; фиксированные ошибки вентилей и shots поддерживаются.
- `aer_density_matrix`: Qiskit Aer с T1/T2, ошибками вентилей и shots.

Для Aer установите `python -m pip install -e ".[aer]"` в окружении ядра.
Градиенты проходят через квантовый блок к классическому энкодеру. Aer вычисляет их
через parameter-shift: с выборочными измерениями в режиме `parameter_shift`,
с точными средними для аналитического Jacobian в остальных режимах.
При `shots=0` все расчёты аналитические. Это не нативный backprop Aer.
Statevector отключает только T1/T2; флажок ошибок вентилей независим.

Шум T1/T2 рассчитывается детерминированно через матрицу плотности. Статистика измерений задаётся отдельно числом shots. Seed обучения отвечает за веса и порядок данных, seed измерений — за выборку результатов.
«Точный» здесь относится к эволюции выбранного канала, а не к гарантированной точности классификации.
`float64` полезен для проверки; `float32` экономит память. Быстродействие измеряется ниже.
""", 'intro')

cell('code', r"""
from pathlib import Path
import os
import sys
ROOT = Path.cwd().resolve()
if not (ROOT / "src" / "qml_beans").is_dir():
    ROOT = ROOT.parent
assert (ROOT / "src" / "qml_beans").is_dir(), "Откройте ноутбук из каталога проекта"
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import json
from dataclasses import replace
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import ipywidgets as w
from IPython.display import display
import matplotlib.pyplot as plt
from qml_beans.model import ModelConfig
from qml_beans.noisy import NoiseConfig, NoisyHybridClassifier, circuit_schedule
from qml_beans.vision_data import load_vision_data, DATASET_CLASSES
from qml_beans.vision_training import (CPUConfig, fit_vision, benchmark_cpu,
                                      report_vision_study, load_noisy_checkpoint)
from qml_beans.notebook import LayerEditor, plot_run, parse_edges
from qml_beans.seeds import parse_seeds
from qml_beans.training import seed_everything
plt.rcParams.update({"font.family": "Times New Roman", "axes.unicode_minus": False})
print("CPU:", os.cpu_count(), "| PyTorch:", torch.__version__)

from qml_beans.measurements import measurement_description
""", 'imports')

cell('markdown', r"""
## Данные и размер изображений

Официальный тест сохраняется отдельно. Валидация выделяется из официальной обучающей
выборки с сохранением долей классов. Подвыборки также стратифицированы.
Изображения уменьшаются билинейно, каналы CIFAR-10 сохраняются; пиксели переводятся
в диапазон [-1, 1]. Никакие статистики по тесту не подбираются.

По умолчанию включён быстрый пробный режим. Снимите флажок для полного датасета.
Размер 14×14 даёт 196 признаков для серых изображений и 588 для цветных.
Для исходного разрешения укажите 28 (MNIST/Fashion) или 32 (CIFAR).
Первая загрузка сохраняет официальные архивы в `data/vision` и проверяет контрольные суммы.

Поле «Сторона, пкс» задаёт высоту и ширину изображения. 14 означает 14×14: 196 признаков для MNIST/Fashion MNIST, 588 для трёх каналов CIFAR-10.
""", 'data-help')

cell('code', r"""
dataset = w.Dropdown(options=list(DATASET_CLASSES), value="Fashion MNIST", description="Датасет:")
image_size = w.BoundedIntText(value=14, min=2, max=32, description="Сторона, пкс:", style={"description_width": "initial"}, tooltip="Изображение будет уменьшено до указанного числа пикселей по каждой стороне: 14 означает 14×14")
small_data = w.Checkbox(value=True, description="Пробная стратифицированная подвыборка")
train_limit = w.BoundedIntText(value=1250, min=10, max=60000, description="Обучение:")
val_limit = w.BoundedIntText(value=250, min=10, max=10000, description="Валидация:")
test_limit = w.BoundedIntText(value=500, min=10, max=10000, description="Тест:")
split_seed = w.IntText(value=42, description="Seed разбиения:", style={"description_width": "initial"})
val_size = w.BoundedFloatText(value=0.15, min=0.01, max=0.5, description="Доля валидации:", style={"description_width": "initial"})
display(w.VBox([dataset, image_size, small_data, w.HBox([train_limit, val_limit, test_limit]), split_seed, val_size]))
""", 'data-controls')

cell('code', r"""
data = load_vision_data(dataset.value, ROOT / "data" / "vision", image_size=image_size.value,
                        seed=split_seed.value, val_size=val_size.value,
                        train_limit=train_limit.value if small_data.value else 0,
                        val_limit=val_limit.value if small_data.value else 0,
                        test_limit=test_limit.value if small_data.value else 0)
print("Признаков:", data.train_x.shape[1], "| Обучение / валидация / тест:",
      len(data.train_y), len(data.val_y), len(data.test_y))
display(data.distribution())
""", 'load-data')

cell('markdown', r"""
## Архитектура и физика шума

Для каждого кубита можно задать одно общее время или восемь значений через запятую.
Все времена **в микросекундах**, требуется $0<T_2\leq 2T_1$.
При нулевой температуре $\rho_{11}(t)=\rho_{11}(0)e^{-t/T_1}$,
$\rho_{01}(t)=\rho_{01}(0)e^{-t/T_2}$.
Это учитывает вклад релаксации в T2: дополнительно применять независимое затухание
с тем же T2 нельзя. Для чистой дефазировки $1/T_\varphi=1/T_2-1/(2T_1)$.

Один слой вращений одного типа на восьми кубитах выполняется параллельно.
Двухкубитные операции выполняются последовательно. После каждого такого временного
интервала шум применяется **ко всем кубитам**, включая простаивающие.
Учитываются кодирование, H, повторная загрузка признаков и задержка перед измерением.
Все RX/RY/RZ используют заданную длительность однокубитной операции.
Это дискретная модель «идеальная операция, затем шум», нулевая температура,
независимые марковские каналы; ошибки считывания, перекрёстные помехи и шум импульса
в эту версию не включены. Параметры по умолчанию — пример, замените их калибровкой устройства.

[Канал PennyLane](https://docs.pennylane.ai/en/stable/code/api/pennylane.ThermalRelaxationError.html).

Для идеального вектора состояния T1/T2 отключаются: релаксация и дефазировка в общем случае создают смешанное состояние, которому нужна матрица плотности. При возврате к шумовому бэкенду прежние T1/T2 и выбор включения шума восстанавливаются.

Редактор задаёт только скрытые слои. Для одного скрытого слоя `(32,)` используется одна активация: первый элемент `("relu", "tanh")`, то есть `relu`; второй элемент этого списка не используется. Активация выхода энкодера `tanh` задаётся отдельно полем `latent_activation` конфигурации модели.


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
""", 'noise-help')

cell('code', r"""
pre_editor = LayerEditor("Слои до квантовой части", (), ())
post_editor = LayerEditor("Слои после квантовой части", (), ())
depth = w.BoundedIntText(value=2, min=1, max=20, description="Квант. слоёв:")
embedding = w.Dropdown(options=["RY", "RX", "RZ"], value="RX", description="Кодирование:")
rotations = w.Text(value="RY,RX", description="Вращения:")
topology = w.Dropdown(options=[ ("Линия", "linear"), ("Два ряда", "brickwork"), ("Три ряда: чётные–нечётные–чётные", "brickwork3"),("Кольцо", "ring"), ("Полная", "full"), ("Без связей", "none"), ("Свои связи", "custom")], description="Связи:")
custom_edges = w.Text(value="[[0,1],[1,2],[2,7]]", description="Свои связи:", layout=w.Layout(width="650px"))
entangler = w.Dropdown(options=["CNOT", "CZ"], description="Вентиль:")
measurement = w.Dropdown(options=["Z", "X", "Y"], description="Измерение:")
hadamard = w.Checkbox(value=False, description="Начальные H")
reupload = w.Checkbox(value=False, description="Повторное кодирование")
backend = w.Dropdown(options=[("PyTorch: матрица плотности", "torch_density"), ("PennyLane: матрица плотности", "pennylane_mixed"), ("PennyLane: вектор состояния", "pennylane_statevector"), ("Qiskit Aer: statevector", "aer_statevector"), ("Qiskit Aer: density matrix", "aer_density_matrix")], layout=w.Layout(width="500px"), description="Бэкенд:")
precision = w.Dropdown(options=["float32", "float64"], description="Точность:")
noise_on = w.Checkbox(value=True, description="Шум T1/T2 включён")
t1 = w.Text(value="40", description="T1, мкс:", layout=w.Layout(width="650px"))
t2 = w.Text(value="30", description="T2, мкс:", layout=w.Layout(width="650px"))
single_time = w.BoundedFloatText(value=0.05, min=0, max=1e6, description="1-куб., мкс:")
two_time = w.BoundedFloatText(value=0.05, min=0, max=1e6, description="2-куб., мкс:")
idle_time = w.BoundedFloatText(value=0.0, min=0, max=1e6, description="Ожидание, мкс:")
display(pre_editor.view, post_editor.view)
display(w.VBox([depth, embedding, rotations, topology, custom_edges, entangler, measurement, hadamard, reupload,
                backend, precision, noise_on, t1, t2, w.HBox([single_time, two_time, idle_time])]))

# При выборе идеального бэкенда значения T1/T2 сохраняются; шум отключается явно.
_previous_noise = {"enabled": noise_on.value}
def sync_backend_controls(change=None):
    ideal = backend.value in ("pennylane_statevector", "aer_statevector")
    if ideal and not noise_on.disabled:
        _previous_noise["enabled"] = noise_on.value
        noise_on.value = False
        noise_on.disabled = True
    elif not ideal and noise_on.disabled:
        noise_on.disabled = False
        noise_on.value = _previous_noise["enabled"]
    for control in (t1, t2, single_time, two_time, idle_time):
        control.disabled = ideal

backend.observe(sync_backend_controls, names="value")
sync_backend_controls()
display(w.HTML("<b>Редакторы задают скрытые слои.</b> Число входных признаков определяется изображением; "
               "выход энкодера всегда содержит 8 параметров углового кодирования. "
               "После измерений получаются 8 признаков; выход классификатора — 10 классов."))

measurement_shots = w.BoundedIntText(value=1000, min=0, max=10000000, description="Измерений (shots):",
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
""", 'model-controls')

cell('markdown', r"""
### Фиксированные ошибки операций (необязательно)

Независимый от T1/T2 переключатель. Задаётся **средняя fidelity вентиля**, а не
вероятность его успешного выполнения. Ошибка калибровки постоянна во всём запуске:
знак генерируется по отдельному seed для каждого типа вентиля и кубита/пары кубитов.
Вращения получают сдвиг угла вокруг своей оси, H — дополнительное вращение Z,
CNOT/CZ — вращение ZZ. Постоянные матрицы и индексы подготавливаются при создании модели.
Это когерентная модель; случайные сбои при каждом применении не моделируются.
Обучение может компенсировать фиксированные сдвиги обучаемых вращений.
Ошибки не генерируются заново между батчами или процессами CPU.
""", 'gate-errors-help')

cell('code', r"""
gate_errors_on = w.Checkbox(value=False, description="Фиксированные ошибки вентилей")
single_fidelity = w.BoundedFloatText(value=99.0, min=100/3, max=100,
    description="Fidelity 1-куб., %:", style={"description_width": "initial"})
two_fidelity = w.BoundedFloatText(value=90.0, min=20, max=100,
    description="Fidelity 2-куб., %:", style={"description_width": "initial"})
gate_error_seed = w.BoundedIntText(value=42, min=0, max=2**32-1,
    description="Seed ошибок:", style={"description_width": "initial"})
display(w.VBox([gate_errors_on, single_fidelity, two_fidelity, gate_error_seed]))
""", 'gate-errors-controls')

cell('markdown', r"""
### Обучаемое кодирование

Угол кодирования: $\pi a_i z_i$, где $z_i$ — выход энкодера, а $a_i$ — отдельный
обучаемый масштаб для каждого из 8 кубитов. Изначально $a_i=1$, то есть схема
совпадает с прежней. Масштабы относятся к квантовой части и замораживаются её
переключателем обучения; при повторном кодировании используются те же 8 масштабов.
Поддерживаются RX, RY и RZ. Выключение опции сохраняет прежнее кодирование.
""", 'embedding-scale-help')

cell('code', r"""
trainable_embedding = w.Checkbox(value=True, description="Обучаемый масштаб кодирования",
    style={"description_width": "initial"}, layout=w.Layout(width="500px"))
display(trainable_embedding)
""", 'embedding-scale-control')

cell('code', r"""
# Обновить классы после изменений кода без перезапуска ядра Jupyter.
import importlib
import qml_beans.model as model_module
import qml_beans.noisy as noisy_module
import qml_beans.vision_training as vision_module

importlib.reload(model_module)
importlib.reload(noisy_module)
importlib.reload(vision_module)
ModelConfig = model_module.ModelConfig
NoiseConfig = noisy_module.NoiseConfig
NoisyHybridClassifier = noisy_module.NoisyHybridClassifier
circuit_schedule = noisy_module.circuit_schedule
CPUConfig = vision_module.CPUConfig
fit_vision = vision_module.fit_vision
benchmark_cpu = vision_module.benchmark_cpu
report_vision_study = vision_module.report_vision_study
load_noisy_checkpoint = vision_module.load_noisy_checkpoint

def times(text):
    values = tuple(float(value.strip()) for value in text.split(","))
    return values * 8 if len(values) == 1 else values

pre_widths, pre_activations = pre_editor.values()
post_widths, post_activations = post_editor.values()
model_config = ModelConfig(n_features=data.train_x.shape[1], n_classes=10,
    pre_layers=pre_widths, pre_activations=pre_activations,
    post_layers=post_widths, post_activations=post_activations,
    quantum_depth=depth.value, embedding=embedding.value, trainable_embedding=trainable_embedding.value,
    rotation_gates=tuple(g.strip().upper() for g in rotations.value.split(",") if g.strip()),
    entanglement=topology.value, custom_edges=parse_edges(custom_edges.value),
    entangler=entangler.value, measurement=measurement.value,
    initial_hadamard=hadamard.value, reupload=reupload.value,
    shots=measurement_shots.value or None, measurement_seed=measurement_seed.value,
    measurement_mode=measurement_mode.value)
noise_config = NoiseConfig(t1_us=times(t1.value), t2_us=times(t2.value),
    single_gate_us=single_time.value, two_gate_us=two_time.value, readout_idle_us=idle_time.value,
    enabled=noise_on.value, backend=backend.value, precision=precision.value,
    gate_errors=gate_errors_on.value, single_gate_fidelity=single_fidelity.value/100,
    two_gate_fidelity=two_fidelity.value/100, gate_error_seed=gate_error_seed.value)
seed_everything(11)
model = NoisyHybridClassifier(model_config, noise_config)
print("Структура энкодера:", " → ".join(map(str, (model_config.n_features, *pre_widths, 8))))
print("После квантовой части:", " → ".join(map(str, (8, *post_widths, model_config.n_classes))))
print("Первое число — входные признаки; в редакторе задаются только скрытые слои.")
print(model.encoder)
print(model.head)
print("Длительность схемы, мкс:", sum(step[2] for step in circuit_schedule(model_config, noise_config)))
if noise_config.backend in ("pennylane_statevector", "aer_statevector"):
    print("Вектор состояния: 256 комплексных амплитуд на объект; T1/T2 отключены.")
else:
    print("Элементов матрицы плотности на объект:", 256 * 256)

print("Измерение:", measurement_description(model_config, backend=noise_config.backend))
""", 'create-model')

cell('markdown', r"""
## Квантовая цепочка

Схема созданной выше модели: 8 кубитов в состоянии $|0\rangle$, без численных углов,
с увеличенными подписями и шрифтом Times New Roman. Обучение не запускается.
По умолчанию показаны вентили; `show_noise=True` также отображает каналы T1/T2,
если шум включён в модели. После изменения архитектуры сначала создайте модель заново.
""", 'circuit-description')

cell('code', r"""
import importlib
import qml_beans.notebook as notebook_helpers

importlib.reload(notebook_helpers)
draw_vision_circuit = notebook_helpers.draw_vision_circuit

output_path = ROOT / "runs" / "quantum_circuit_vision.png"
figure, _ = draw_vision_circuit(model, output_path, show_noise=False)
display(figure)
print(f"Сохранено: {output_path}")
plt.close(figure)
""", 'draw-circuit')

cell('markdown', r"""
## CPU: процессы, микробатчи и общий шаг оптимизатора

`Размер батча` — **общее** число объектов на один шаг Adam независимо от числа процессов.
В режиме «Части батча» общий батч делится между процессами. В режиме «Очередь микробатчей»
процессы получают несколько мелких задач из того же общего батча.
Во всех случаях градиенты суммируются по объектам и делятся на общий размер батча.
Один шаг Adam выполняется после завершения всех задач. При неравных частях учитывается
число объектов, а не число процессов. Это сохраняет математический смысл обычного обучения.

Каждый процесс обрабатывает данные микробатчами и сразу освобождает граф после backward.
Веса и данные передаются через общую память CPU, работники читают веса, основной процесс
изменяет их только после барьера. Функции процессов вынесены в импортируемый модуль,
что позволяет запуск из Jupyter под Windows.

Начните с 1–2 процессов, 1 потока и микробатча 1–2. Число процессов × потоки не должно
превышать доступные CPU. Память под графы растёт с числом процессов и размером микробатча;
матрица 256×256 — лишь состояние, backward требует гораздо больше памяти.
Ускорение не гарантируется: на короткой схеме затраты IPC могут перевесить выигрыш.
Сначала измерьте время на своём CPU. Локальные процессы используют одно физическое устройство;
это не распределение по нескольким компьютерам.
""", 'cpu-help')

cell('code', r"""
workers = w.BoundedIntText(value=16, min=1, max=os.cpu_count() or 1, description="Процессов:")
threads = w.BoundedIntText(value=1, min=1, max=os.cpu_count() or 1, description="Потоков/проц.:")
microbatch = w.BoundedIntText(value=2, min=1, max=256, description="Микробатч:")
parallel_mode = w.Dropdown(options=[("Части батча", "split_batch"), ("Очередь микробатчей", "batch_queue")], description="Режим:")
batch_size = w.BoundedIntText(value=96, min=1, max=4096, description="Размер батча:")
epochs = w.BoundedIntText(value=100, min=1, max=10000, description="Эпох:")
patience = w.BoundedIntText(value=20, min=0, max=10000, description="Терпение:")
min_delta = w.BoundedFloatText(value=0.0, min=0.0, max=100, description="Мин. улучшение:")
learning_rate = w.FloatText(value=1e-3, description="Шаг классич.:")
quantum_rate = w.FloatText(value=1e-3, description="Шаг квант.:")
use_quantum = w.Checkbox(value=True, description="Квантовая часть включена")
train_classical = w.Checkbox(value=True, description="Обучать классическую часть")
train_quantum = w.Checkbox(value=True, description="Обучать квантовую часть")
seeds = w.Text(value="11", description="Seed обучения:", style={"description_width": "initial"}, tooltip="Числа для инициализации весов и перемешивания данных. Например: 11,22,33. Разбиение задаётся отдельно")
output_name = w.Text(value="noisy_vision", description="Имя серии:")
display(w.VBox([w.HBox([workers, threads, microbatch]), parallel_mode, batch_size, epochs,
                patience, min_delta, learning_rate, quantum_rate, use_quantum, train_classical,
                train_quantum, seeds, output_name]))
def current_cpu():
    return CPUConfig(workers=workers.value, threads_per_worker=threads.value,
                     microbatch_size=microbatch.value, parallel_mode=parallel_mode.value)
""", 'training-controls')

cell('markdown', r"""
### Измерение скорости (необязательно)

Сравниваются 1 процесс и выбранное число процессов на одинаковом батче и весах.
Измеряется forward + backward; создание процессов и прогрев исключены, шаг Adam не выполняется.
Оба бэкенда можно сравнить переключением бэкенда и повторным созданием модели.
Рекомендуется повторять измерения на достаточно большом батче. Это отдельная проверка скорости,
а не прогноз качества классификации.
""", 'benchmark-help')

cell('code', r"""
run_benchmark = False  # True для измерения; обучение не запускается
if run_benchmark:
    model.set_mode(quantum=use_quantum.value, train_classical=train_classical.value, train_quantum=train_quantum.value)
    configurations = [replace(current_cpu(), workers=1)]
    if workers.value > 1:
        configurations.append(current_cpu())
    count = min(batch_size.value, len(data.train_y))
    benchmark = benchmark_cpu(model, data.train_x[:count], data.train_y[:count], configurations)
    display(benchmark.rename(columns={"workers": "Процессы", "threads_per_worker": "Потоки",
        "microbatch_size": "Микробатч", "parallel_mode": "Режим", "seconds": "Секунд на батч",
        "samples_per_second": "Объектов в секунду", "backend": "Бэкенд", "precision": "Точность"}))
""", 'benchmark')

cell('markdown', r"""
## Обучение

Ранняя остановка настраивается выше: валидационная кросс-энтропия должна уменьшаться
не менее чем на «Мин. улучшение»; «Терпение» задаёт число эпох без улучшения,
0 отключает раннюю остановку. Максимум задаёт «Эпох».
Тест оценивается после восстановления лучших весов, выбранных только по валидации.

При нескольких начальных значениях модель каждый раз инициализируется заново;
разбиение фиксировано. При одном начальном значении можно продолжать текущую модель:
включите `continue_current_model`, например после этапа без квантовой части.
Оптимизатор при новом этапе создаётся заново. Настройки заморозки применяются ко всему этапу.
Для повторного этапа создаётся новый каталог, имеющиеся веса и отчёты не перезаписываются.

**Seed обучения** — целое число для воспроизводимой инициализации весов и перемешивания обучающих объектов. `11,22,33` запускает три инициализации. **Seed разбиения** задаётся отдельно и фиксирует состав обучающей/валидационной выборок.
""", 'train-help')

cell('code', r"""
continue_current_model = False
selected_seeds = parse_seeds(seeds.value)
if continue_current_model and len(selected_seeds) != 1:
    raise ValueError("Продолжение текущей модели допускает одно начальное значение")
if Path(output_name.value).name != output_name.value or output_name.value in ("", ".", ".."):
    raise ValueError("Имя серии должно быть простым именем каталога")
study_folder = ROOT / "runs" / output_name.value / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
results = []
for seed in selected_seeds:
    if not continue_current_model:
        seed_everything(seed)
        model = NoisyHybridClassifier(model_config, noise_config)
    model.set_mode(quantum=use_quantum.value, train_classical=train_classical.value, train_quantum=train_quantum.value)
    result = fit_vision(model, data, study_folder / f"seed_{seed}", cpu=current_cpu(),
        epochs=epochs.value, batch_size=batch_size.value, patience=patience.value, min_delta=min_delta.value,
        lr=learning_rate.value, quantum_lr=quantum_rate.value, seed=seed,
        initialization="continued" if continue_current_model else "independent")
    results.append(result)
    plot_run(result)
print("Результаты:", study_folder)
""", 'train')

cell('markdown', r"""
## Сбор начальных значений и графики разброса

Повторно выполнять обучение для этого раздела не нужно. Для объединения отдельных запусков
задайте список папок `seed_*` в `collect_paths`. Сборщик отклоняет повторные значения
инициализации, разные настройки модели, шума, данных или обучения.
Здесь доступны среднее ± выборочное стандартное отклонение (`sd`) или приближённый
95%-й доверительный интервал Стьюдента среднего по независимым инициализациям (`ci95`).
ДИ условен на фиксированном датасете и разбиении; при одном запуске разброс не определяется.
Для продолжаемых моделей сбор разных историй предварительного обучения не допускается.
""", 'report-help')

cell('code', r"""
collect_paths = [Path(result["output"]) for result in results]
band = "sd"  # либо "ci95"
summary = report_vision_study(collect_paths, study_folder / "report", band=band)
display(summary)
""", 'report')

cell('markdown', r"""
## Восстановление весов

Выполните следующую ячейку с непустым путём, чтобы восстановить модель и настройки.
Архитектура и шум восстанавливаются из checkpoint, переключатели обучения синхронизируются.
Для продолжения используйте `continue_current_model=True` и подходящие данные из того же протокола.

Источники данных: [MNIST](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py),
[Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist),
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).
""", 'restore-help')

cell('code', r"""
checkpoint_path = ""
if checkpoint_path:
    model, checkpoint = load_noisy_checkpoint(checkpoint_path)
    model_config, noise_config = model.config, model.noise
    gate_errors_on.value = noise_config.gate_errors
    single_fidelity.value = 100 * noise_config.single_gate_fidelity
    two_fidelity.value = 100 * noise_config.two_gate_fidelity
    gate_error_seed.value = noise_config.gate_error_seed
    measurement_mode.value = model_config.measurement_mode
    measurement_shots.value = model_config.shots or 0
    measurement_seed.value = model_config.measurement_seed
    trainable_embedding.value = model_config.trainable_embedding
    use_quantum.value = model.use_quantum
    train_classical.value = model.train_classical
    train_quantum.value = model.train_quantum
    print("Восстановлена эпоха:", checkpoint["epoch"])
""", 'restore')


if __name__ == "__main__":
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells,
                    metadata=dict(kernelspec=dict(display_name="Python (qml-beans)", language="python", name="qml-beans"),
                                  language_info=dict(name="python", version="3.12")))
    path = Path(__file__).resolve().parents[1] / "notebooks" / "noisy_vision_cpu.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(path)
