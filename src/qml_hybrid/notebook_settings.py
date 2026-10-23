"""Copy/save notebook controls and portable classical architecture as JSON."""

import json
from pathlib import Path

import torch

from .classical import normalize_spec


def _controls(namespace):
    import ipywidgets as w
    types = (w.IntText, w.BoundedIntText, w.FloatText, w.BoundedFloatText,
             w.Text, w.Textarea, w.Dropdown, w.Checkbox)
    return {name: value for name, value in namespace.items()
            if not name.startswith("_") and isinstance(value, types)
            and not name.startswith("settings_")}


def export_notebook_settings(namespace):
    """Capture current values, not stale model_config and not weights or results."""
    source = namespace.get("classical_source", "widgets")
    architecture = None
    if source == "code":
        # Building a description must not change subsequent model initialization.
        with torch.random.fork_rng(devices=[]):
            architecture = namespace["classical_from_code"]()
    elif source == "saved":
        architecture = namespace["classical_saved_config"]
    elif source != "widgets":
        raise ValueError("classical_source: widgets, code или saved")
    editors = {}
    for name in ("pre_editor", "post_editor"):
        if name in namespace:
            widths, activations = namespace[name].values()
            editors[name] = dict(widths=list(widths), activations=list(activations))
    return dict(format="qml_hybrid_notebook_settings_v1",
                widgets={name: widget.value for name, widget in _controls(namespace).items()},
                editors=editors,
                classical=dict(source="code" if architecture is not None else "widgets", config=architecture))


def apply_notebook_settings(settings, namespace):
    """Apply controls only. Data/model must then be rebuilt explicitly."""
    from traitlets import TraitError
    if settings.get("format") != "qml_hybrid_notebook_settings_v1":
        raise ValueError("Ожидается конфигурация qml_hybrid_notebook_settings_v1")
    controls = _controls(namespace)
    values = settings["widgets"]
    for name, value in values.items():
        if name not in controls:
            raise ValueError(f"Сначала выполните ячейки виджетов: отсутствует {name}")
        widget = controls[name]
        if hasattr(widget, "min") and not widget.min <= value <= widget.max:
            raise ValueError(f"{name}: значение {value} вне границ [{widget.min}, {widget.max}]")
        try:
            widget.traits()["value"]._validate(widget, value)
        except TraitError as exc:
            raise ValueError(f"{name}: {exc}") from exc
        if hasattr(widget, "options"):
            options = [option[1] if isinstance(option, (list, tuple)) else option for option in widget.options]
            if value not in options:
                raise ValueError(f"{name}: недопустимое значение {value!r}")
    classical = settings.get("classical", dict(source="widgets", config=None))
    architecture = classical.get("config")
    if architecture is not None:
        if not {"encoder_spec", "head_spec", "image_shape"} <= set(architecture) or set(architecture) - {"encoder_spec", "head_spec", "image_shape", "latent_features"}:
            raise ValueError("Некорректные поля классической архитектуры")
        width = architecture.get("latent_features", 8)
        if type(width) is not int or width < 1:
            raise ValueError("latent_features: положительное целое")
        normalize_spec(architecture["encoder_spec"])
        normalize_spec(architecture["head_spec"])
    for name, editor in settings.get("editors", {}).items():
        if name not in ("pre_editor", "post_editor") or name not in namespace:
            raise ValueError(f"Редактор отсутствует: {name}")
        namespace[name].validate_values(editor["widths"], editor["activations"])
    # Dataset changes trigger split-policy callbacks; restore it first.
    if "dataset" in values:
        controls["dataset"].value = values["dataset"]
    for name, value in values.items():
        if name != "dataset":
            controls[name].value = value
    if namespace.get("dataset") is not None and namespace["dataset"].value == "CIFAR-10":
        if "use_validation" in namespace:
            namespace["use_validation"].value = False
    for name, editor in settings.get("editors", {}).items():
        namespace[name].set_values(editor["widths"], editor["activations"])
    namespace["classical_source"] = "saved" if architecture is not None else "widgets"
    namespace["classical_saved_config"] = architecture


def settings_panel(namespace, root):
    """JSON text is selectable/copyable; file actions require explicit button clicks."""
    import ipywidgets as w
    from IPython.display import display

    path = w.Text(value=str(Path(root) / "configs" / "vision_settings.json"),
                  description="Файл JSON:", layout=w.Layout(width="95%"))
    text = w.Textarea(description="JSON:", layout=w.Layout(width="95%", height="300px"))
    output = w.Output()
    capture, apply, save, load = [w.Button(description=label) for label in
                                ("Собрать настройки", "Применить JSON", "Сохранить JSON", "Загрузить JSON")]

    def action(kind):
        with output:
            output.clear_output()
            try:
                if kind in ("capture", "save"):
                    text.value = json.dumps(export_notebook_settings(namespace), ensure_ascii=False, indent=2)
                if kind == "save":
                    destination = Path(path.value).expanduser()
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text(text.value + "\n", encoding="utf-8")
                    print(f"Настройки сохранены: {destination}")
                if kind == "load":
                    text.value = Path(path.value).expanduser().read_text(encoding="utf-8")
                if kind in ("apply", "load"):
                    apply_notebook_settings(json.loads(text.value), namespace)
                    print("Настройки применены. Выполните загрузку данных и создание модели заново. "
                          "Ячейку определения классики кодом повторять не нужно.")
            except (ValueError, TypeError, KeyError, OSError) as exc:
                print(f"Ошибка: {exc}")

    capture.on_click(lambda _: action("capture"))
    apply.on_click(lambda _: action("apply"))
    save.on_click(lambda _: action("save"))
    load.on_click(lambda _: action("load"))
    panel = w.VBox([w.HTML("<b>Настройки эксперимента: копирование и JSON</b>"), path,
                    w.HBox([capture, apply, save, load]), text, output])
    display(panel)
    return panel
