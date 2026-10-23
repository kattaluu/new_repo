# qml_hybrid — cpp

`notebooks/noisy_vision_cpu.ipynb`: гибридное обучение с 8 кубитами,
настройкой архитектуры, T1/T2, ошибками вентилей, shots и CPU-процессами.
Бэкенды: PyTorch, PennyLane, CustomDevice C++ (SV и матрица плотности).

## Установка

Python 3.11+; для сборки C++ нужен компилятор с поддержкой C++17.
Windows: Visual Studio Build Tools, компонент «Разработка классических приложений на C++».
Linux/macOS: GCC/Clang и инструменты сборки.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[notebook,dev]"
.venv\Scripts\python -m pip install "pybind11==3.0.1" wheel
.venv\Scripts\python native/custom_device/setup.py build_ext --inplace
.venv\Scripts\python -m jupyterlab notebooks/noisy_vision_cpu.ipynb
```

Linux/macOS: вместо `.venv\Scripts\python` используйте `.venv/bin/python`.
Выберите ядро созданного окружения. В поле `CustomDevice` уже указан
`<корень репозитория>/native/custom_device`; сборка создаёт там модули
`engine8q_dm` и `engine8q_sv` для текущей ОС и версии Python.
Для C++ SV отключается T1/T2; для C++ DM шум доступен. Внутренняя точность C++ — complex128.
PyTorch и PennyLane можно использовать без сборки C++.

## Запуск

Выполните ячейки импорта, настройки и загрузки данных, создания модели и обучения.
Классическую архитектуру можно задать виджетами или кодом. Число CPU-процессов
и размер микробатча подберите под ресурсы компьютера. QPU time est сохранён.

Данные загружаются в `data/vision`; результаты сохраняются в `runs/`.
В репозиторий не включены датасеты, веса, результаты и готовые бинарные файлы.
Чтобы перестроить старые графики, положите свои серии в `runs/noisy_vision/`
и измените пути/seed в соответствующей ячейке. Обучение для этого не требуется.

## Проверка

```powershell
.venv\Scripts\python -m pytest tests -q
```

C++-тесты требуют собранных движков, иначе соответствующие проверки пропускаются.
Исходники `native/custom_device/engine_*.cpp` перенесены из рабочего CustomDevice;
устаревшие обёртки PennyLane 0.25 не используются.
