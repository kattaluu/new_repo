"""Synchronous CPU data parallelism, exact microbatch accumulation and image experiments.

Spawn targets live in this importable module, so Windows Jupyter does not pickle cell functions.
Workers only read shared weights. The parent owns Adam and updates AFTER the worker barrier.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from .model import ModelConfig, saved_model_config
from .noisy import NoiseConfig, NoisyHybridClassifier
from .regularization import RegularizationConfig, regularization_penalty
from .measurements import classification_loss
from .seeds import _plot_seed_report, sample_statistics, summarize_seed_runs
from .training import classification_metrics, seed_everything, write_json
from .batchnorm import collect_batchnorm_moments, snapshot_batchnorm, update_batchnorm


@dataclass(frozen=True)
class CPUConfig:
    workers: int = 1
    threads_per_worker: int = 1
    microbatch_size: int = 2
    # split_batch: one shard per process; batch_queue: several independent small tasks.
    parallel_mode: str = "split_batch"

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in (self.workers, self.threads_per_worker, self.microbatch_size)):
            raise ValueError("Число процессов, потоков и размер микробатча должны быть положительными целыми")
        if self.parallel_mode not in ("split_batch", "batch_queue"):
            raise ValueError("parallel_mode: split_batch или batch_queue")


def _compute(model, x, y, microbatch, backward):
    model.zero_grad(set_to_none=True)
    predictions, loss_sum = [], 0.0
    enabled = backward and any(p.requires_grad for p in model.parameters())
    with collect_batchnorm_moments(model) as statistics, (torch.enable_grad() if enabled else torch.no_grad()):
        for start in range(0, len(y), microbatch):
            logits = model(x[start:start + microbatch])
            loss = classification_loss(model, logits, y[start:start + microbatch], reduction="sum")
            if enabled:
                loss.backward()
            loss_sum += float(loss.detach())
            predictions.append(logits.detach())
    gradients = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()] if backward else []
    return gradients, loss_sum, torch.cat(predictions), statistics


def _init_worker(model_config, noise_config, mode, shared_state, tensors, cpu):
    global _WORKER_MODEL, _SHARED_STATE, _TENSORS, _CPU, _THREAD_LIMIT
    torch.set_num_threads(cpu.threads_per_worker)
    torch.set_num_interop_threads(1)
    from threadpoolctl import threadpool_limits
    _THREAD_LIMIT = threadpool_limits(limits=cpu.threads_per_worker)
    _WORKER_MODEL = NoisyHybridClassifier(ModelConfig(**model_config), NoiseConfig(**noise_config))
    _WORKER_MODEL.set_mode(**mode)
    _SHARED_STATE, _TENSORS, _CPU = shared_state, tensors, cpu


def _worker_task(task):
    split, indices, backward, training, measurement_seed = task
    _WORKER_MODEL.load_state_dict(_SHARED_STATE)
    _WORKER_MODEL.train(training)
    if measurement_seed is not None:
        _WORKER_MODEL.quantum.set_measurement_seed(measurement_seed)
    x, y = _TENSORS[split]
    return _compute(_WORKER_MODEL, x[indices], y[indices], _CPU.microbatch_size, backward)


class CPUEngine:
    def __init__(self, model, tensors, cpu):
        self.model, self.cpu = model, cpu
        dtype = next(model.parameters()).dtype
        self.tensors = {name: (x.to(dtype=dtype).contiguous(), y.long().contiguous())
                        for name, (x, y) in tensors.items()}
        self.pool = None
        self.measurement_step = 0

    def __enter__(self):
        self.old_threads = torch.get_num_threads()
        torch.set_num_threads(self.cpu.threads_per_worker)
        if self.cpu.workers > 1:
            self.model.share_memory()
            for x, y in self.tensors.values():
                x.share_memory_()
                y.share_memory_()
            self.pool = ProcessPoolExecutor(
                max_workers=self.cpu.workers, mp_context=mp.get_context("spawn"),
                initializer=_init_worker,
                initargs=(self.model.config_dict(), self.model.noise_dict(), self.model.mode_dict(),
                          self.model.state_dict(), self.tensors, self.cpu))
        return self

    def __exit__(self, *exc):
        if self.pool:
            self.pool.shutdown(wait=True, cancel_futures=True)
        torch.set_num_threads(self.old_threads)

    def run(self, split, indices, *, backward=False, training=None):
        indices = torch.as_tensor(indices, dtype=torch.long)
        if len(indices) == 0:
            raise ValueError("Пустой батч")
        training = backward if training is None else training
        self.model.train(training)
        previous_bn = snapshot_batchnorm(self.model)
        if self.cpu.parallel_mode == "split_batch":
            chunks = [part for part in torch.tensor_split(indices, self.cpu.workers) if len(part)]
        else:
            chunks = list(indices.split(self.cpu.microbatch_size))
        self.measurement_step += 1
        seeds = [int(np.random.SeedSequence([self.model.config.measurement_seed, self.measurement_step,
                                             {"train": 0, "val": 1, "test": 2}[split], index]).generate_state(1)[0])
                 if self.model.config.shots is not None else None for index in range(len(chunks))]
        if self.pool:
            # map is ordered and constitutes a full barrier before the next optimizer step.
            results = list(self.pool.map(_worker_task, [(split, part, backward, training, seed) for part, seed in zip(chunks, seeds)]))
        else:
            x, y = self.tensors[split]
            results = []
            for part, seed in zip(chunks, seeds):
                if seed is not None:
                    self.model.quantum.set_measurement_seed(seed)
                results.append(_compute(self.model, x[part], y[part], self.cpu.microbatch_size, backward))
        update_batchnorm(self.model, previous_bn, [result[3] for result in results])
        if backward:
            for index, param in enumerate(self.model.parameters()):
                values = [r[0][index] for r in results if r[0][index] is not None]
                param.grad = sum(values) / len(indices) if values else None
        return sum(r[1] for r in results) / len(indices), torch.cat([r[2] for r in results])


def _tensors(data):
    return {name: (getattr(data, name + "_x"), getattr(data, name + "_y"))
            for name in ("train", "val", "test")}


def _metrics(engine, split, classes, batch_size):
    x, y = engine.tensors[split]
    logits = [engine.run(split, torch.arange(start, min(start + batch_size, len(y))))[1]
              for start in range(0, len(x), batch_size)]
    return classification_metrics(torch.cat(logits), y.numpy(), classes)


def load_noisy_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != "noisy_vision_v1":
        raise ValueError("Ожидался checkpoint шумовой модели изображений")
    model = NoisyHybridClassifier(saved_model_config(saved["model_config"]), NoiseConfig(**saved["noise_config"]))
    model.load_state_dict(saved["state_dict"])
    model.set_mode(**saved["mode"])
    return model, saved


def fit_vision(model, data, output, *, cpu=CPUConfig(), epochs=10, batch_size=32,
               lr=1e-3, quantum_lr=1e-3, patience=5, min_delta=0.0, seed=42,
               initialization="independent", regularization=RegularizationConfig()):
    if isinstance(regularization, dict):
        regularization = RegularizationConfig(**regularization)
    if not isinstance(regularization, RegularizationConfig):
        raise ValueError("regularization: нужен RegularizationConfig или словарь коэффициентов")
    if epochs < 1 or batch_size < 1 or patience < 0 or min_delta < 0 or lr <= 0 or quantum_lr <= 0:
        raise ValueError("Некорректные настройки обучения")
    if initialization not in ("independent", "continued"):
        raise ValueError("initialization: independent или continued")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "best.pt").exists() or (output / "history.json").exists():
        raise FileExistsError(f"Папка уже содержит обучение: {output}. Укажите новую папку.")
    seed_everything(seed)
    params = [dict(params=[p for name, p in model.named_parameters() if not name.startswith("quantum.") and p.requires_grad], lr=lr),
              dict(params=[p for p in model.quantum.parameters() if p.requires_grad], lr=quantum_lr)]
    params = [group for group in params if group["params"]]
    optimizer = torch.optim.Adam(params) if params else None
    settings = dict(cpu=asdict(cpu), epochs=epochs, batch_size=batch_size, lr=lr, quantum_lr=quantum_lr,
                    patience=patience, min_delta=min_delta, seed=seed, initialization=initialization,
                    regularization=asdict(regularization))
    if any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules()):
        settings["batchnorm"] = dict(normalization="local_microbatch", running_statistics="pooled_per_optimizer_batch",
                                     singleton="running_statistics", frozen_classical="eval")
        print("BatchNorm: нормализация по микробатчу; статистики оценки объединяются по батчу. "
              "При одном значении на канал используются накопленные статистики.", flush=True)
    if initialization == "continued":
        digest = hashlib.sha256()
        for value in model.state_dict().values():
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        settings["initial_weights_sha256"] = digest.hexdigest()
    history, best_loss, best_epoch, bad_epochs = [], float("inf"), 0, 0
    generator = torch.Generator().manual_seed(seed)
    has_validation = bool(len(data.val_y))
    settings["selection"] = "validation_loss" if has_validation else "last_epoch"
    settings["test_evaluation"] = "final_only" if has_validation else "each_epoch"
    if not has_validation:
        print("Без валидации: ранняя остановка отключена, используются веса последней эпохи. "
              "Тест оценивается после каждой эпохи.", flush=True)

    def save(path, epoch):
        torch.save(dict(format="noisy_vision_v1", model_config=model.config_dict(), noise_config=model.noise_dict(),
                        mode=model.mode_dict(), state_dict=model.state_dict(), optimizer=optimizer.state_dict() if optimizer else None,
                        epoch=epoch, history=history, settings=settings, data=data.metadata), path)

    with CPUEngine(model, _tensors(data), cpu) as engine:
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            order = torch.randperm(len(data.train_y), generator=generator)
            losses, penalties, correct = 0.0, 0.0, 0
            for indices in order.split(batch_size):
                loss, logits = engine.run("train", indices, backward=optimizer is not None, training=True)
                penalty = regularization_penalty(model, regularization)
                if not torch.isfinite(penalty):
                    raise FloatingPointError("Нечисловой штраф регуляризации")
                if optimizer:
                    if penalty.requires_grad:
                        penalty.backward()
                    optimizer.step()
                losses += loss * len(indices)
                penalties += float(penalty.detach()) * len(indices)
                correct += int((logits.argmax(-1) == data.train_y[indices]).sum())
            validation = _metrics(engine, "val", data.classes, batch_size) if has_validation else None
            epoch_test = None if has_validation else _metrics(engine, "test", data.classes, batch_size)
            if (not np.isfinite(losses)
                    or (validation is not None and not np.isfinite(validation["loss"]))
                    or (epoch_test is not None and not np.isfinite(epoch_test["loss"]))):
                raise FloatingPointError("Обнаружена нечисловая функция потерь")
            history.append(dict(epoch=epoch, train_loss=losses / len(order), train_accuracy=correct / len(order),
                                regularization_loss=penalties / len(order),
                                train_objective=(losses + penalties) / len(order),
                                seconds=time.perf_counter() - started))
            if validation is not None:
                history[-1].update(val_loss=validation["loss"], val_accuracy=validation["accuracy"])
            if epoch_test is not None:
                history[-1].update(test_loss=epoch_test["loss"], test_accuracy=epoch_test["accuracy"])
            if not has_validation or validation["loss"] < best_loss - min_delta:
                best_loss, best_epoch, bad_epochs = validation["loss"] if has_validation else float("inf"), epoch, 0
                save(output / "best.pt", epoch)
            else:
                bad_epochs += 1
            write_json(output / "history.json", history)
            pd.DataFrame(history).to_csv(output / "history.csv", index=False)
            validation_note = f"валидация={validation['accuracy']:.4f}; " if validation is not None else ""
            if epoch_test is not None:
                validation_note = f"тест: потери={epoch_test['loss']:.4f}, точность={epoch_test['accuracy']:.4f}; "
            print(f"Эпоха {epoch}/{epochs}: потери={history[-1]['train_loss']:.4f}; "
                  f"{validation_note}время={history[-1]['seconds']:.1f} с", flush=True)
            if has_validation and patience and bad_epochs >= patience:
                break
        save(output / "last.pt", epoch)
        best, _ = load_noisy_checkpoint(output / "best.pt")
        model.load_state_dict(best.state_dict())
        # Shared parent storage is retained by load_state_dict; workers see best weights.
        validation = _metrics(engine, "val", data.classes, batch_size) if has_validation else None
        test = _metrics(engine, "test", data.classes, batch_size) if has_validation else epoch_test
    result = dict(output=str(output), metrics=dict(test=test, validation=validation, best_epoch=best_epoch,
                  classes=list(data.classes), mode=model.mode_dict(), settings=settings,
                  model_config=model.config_dict(), noise_config=model.noise_dict(), data=data.metadata))
    write_json(output / "metrics.json", result["metrics"])
    pd.DataFrame(history).to_csv(output / "history.csv", index=False)
    epoch_label = "лучшая эпоха" if has_validation else "последняя эпоха"
    print(f"Тест: точность={test['accuracy']:.4f}; макро-F1={test['macro_f1']:.4f}; {epoch_label}={best_epoch}", flush=True)
    return result


def benchmark_cpu(model, x, y, configurations, *, repeats=2):
    """Forward + backward, no optimizer step. Initialization excluded; weights unchanged."""
    records = []
    state = {name: value.clone() for name, value in model.state_dict().items()}
    for cpu in configurations:
        model.load_state_dict(state)
        with CPUEngine(model, {"train": (x, y)}, cpu) as engine:
            indices = torch.arange(len(y))
            engine.run("train", indices, backward=True)  # warm-up (includes process creation)
            started = time.perf_counter()
            for _ in range(repeats):
                engine.run("train", indices, backward=True)
            seconds = (time.perf_counter() - started) / repeats
        records.append(dict(**asdict(cpu), seconds=seconds, samples_per_second=len(y) / seconds,
                            backend=model.noise.backend, precision=model.noise.precision))
    model.zero_grad(set_to_none=True)
    model.load_state_dict(state)
    assert all(torch.equal(state[name], value) for name, value in model.state_dict().items())
    return pd.DataFrame(records)


def _protocol_differences(reference, current, *, limit=20):
    """Bounded field-level diagnostics, including nested architecture and split indices."""
    missing = object()
    differences = []

    def preview(value):
        if value is missing:
            return "<отсутствует>"
        text = repr(value)
        return text if len(text) <= 140 else text[:137] + "..."

    def visit(left, right, field):
        if len(differences) >= limit or left == right:
            return
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(left.keys() | right.keys()):
                visit(left.get(key, missing), right.get(key, missing), f"{field}.{key}" if field else key)
        elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
            if len(left) != len(right):
                differences.append(f"{field}: длина {len(left)} != {len(right)}")
            for index, (a, b) in enumerate(zip(left, right)):
                visit(a, b, f"{field}[{index}]")
                if len(differences) >= limit:
                    break
        else:
            differences.append(f"{field}: {preview(left)} != {preview(right)}")

    visit(reference, current, "")
    return differences


def collect_vision_runs(paths, *, seeds=None, allow_mixed_protocols=False):
    """Read and validate a homogeneous seed group without training or plotting."""
    selected_seeds = None
    if seeds is not None:
        seeds = list(seeds)
        if not seeds or any(isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) for seed in seeds):
            raise ValueError("seeds: непустой список целых начальных значений или None для всех запусков")
        selected_seeds = {int(seed) for seed in seeds}
        if len(selected_seeds) != len(seeds):
            raise ValueError("seeds не должен содержать повторные начальные значения")
    runs = []
    for path in paths:
        path = Path(path)
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        if selected_seeds is None or metrics["settings"]["seed"] in selected_seeds:
            runs.append((path, metrics))
    if selected_seeds is not None:
        missing = selected_seeds - {metrics["settings"]["seed"] for _, metrics in runs}
        if missing:
            raise ValueError(f"В переданных папках не найдены seed: {sorted(missing)}")
    rows, seen_seeds, seed_paths = [], set(), {}
    protocols, protocol_differences = [], []
    reference_protocol = reference_path = reference_seed = None
    for path, metrics in runs:
        if len(runs) > 1 and metrics["settings"].get("initialization") == "continued":
            raise ValueError("Продолжаемые модели нельзя объявлять независимыми инициализациями")
        settings = {k: v for k, v in metrics["settings"].items() if k not in ("seed", "cpu")}
        settings["regularization"] = asdict(RegularizationConfig(**settings.get("regularization", {})))
        settings.setdefault("selection", "validation_loss" if metrics.get("validation") is not None else "last_epoch")
        settings.setdefault("test_evaluation", "final_only")
        protocol = dict(model=asdict(saved_model_config(metrics["model_config"])), noise=asdict(NoiseConfig(**metrics["noise_config"])), mode=metrics["mode"],
                        data=metrics["data"], settings=settings)
        seed = metrics["settings"]["seed"]
        if seed in seed_paths:
            raise ValueError(f"Повторный seed={seed}:\n  {seed_paths[seed]}\n  {path}\n"
                             "Один seed нельзя учитывать дважды как независимый запуск.")
        if reference_protocol is None:
            reference_protocol, reference_path, reference_seed = protocol, path, seed
        else:
            differences = _protocol_differences(reference_protocol, protocol)
            if differences and not allow_mixed_protocols:
                raise ValueError(f"Различаются протоколы запусков:\n"
                                 f"  эталон: seed={reference_seed}, {reference_path}\n"
                                 f"  сравниваемый: seed={seed}, {path}\n"
                                 + "\n".join("  - " + item for item in differences)
                                 + "\nРазные папки допустимы; объединяются совпадающие настройки и разные seed.")
            if differences:
                protocol_differences.append(dict(seed=seed, reference_seed=reference_seed, fields=differences))
        protocols.append(dict(seed=seed, path=str(path), protocol=protocol))
        seed_paths[seed] = path
        seen_seeds.add(seed)
        rows.append(dict(seed=seed, stage="hybrid" if metrics["mode"]["quantum"] else "classical",
                         stage_index=0, output=str(path),
                         **{f"test_{key}": metrics["test"][key] for key in ("accuracy", "macro_f1", "balanced_accuracy", "loss")},
                         history=json.loads((path / "history.json").read_text(encoding="utf-8"))))
        if metrics.get("validation") is not None:
            rows[-1].update(val_accuracy=metrics["validation"]["accuracy"],
                            val_macro_f1=metrics["validation"]["macro_f1"])
    if not rows:
        raise ValueError("Нет результатов для сбора")
    if protocol_differences:
        protocol = dict(mixed_protocols=True, runs=protocols, differences=protocol_differences)
    return rows, protocol, sorted(seen_seeds)


def report_vision_study(paths, output, *, band="sd", show=True, seeds=None, legend_loc="lower right", legend_on="right",
                        legend_ncol=1, allow_mixed_protocols=False,
                        panel_positions=((-0.12, 1.04), (-0.12, 1.04))):
    """Report all runs or only the initialization seeds explicitly selected by ``seeds``."""
    if band not in ("sd", "ci95"):
        raise ValueError("band: sd или ci95")
    rows, protocol, seen_seeds = collect_vision_runs(paths, seeds=seeds, allow_mixed_protocols=allow_mixed_protocols)
    mixed = protocol.get("mixed_protocols", False)
    if mixed and band != "sd":
        raise ValueError("Для смешанной серии используйте band='sd': разброс включает различия настроек.")
    config = dict(split_policy="vary" if mixed else "fixed", mixed_protocols=mixed,
                  stages=[dict(name=stage) for stage in dict.fromkeys(row["stage"] for row in rows)])
    summary = summarize_seed_runs(config, rows)
    has_validation = "val_accuracy" in rows[0]
    if has_validation:
        summary.append(dict(stage=rows[0]["stage"], metric="val_macro_f1",
                            **sample_statistics([row["val_macro_f1"] for row in rows], allow_ci=not mixed)))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", dict(protocol=protocol, statistics=summary, band=band,
               selected_seeds=sorted(seen_seeds),
               mixed_protocols=mixed,
               interpretation=("Смешанная серия: среднее и SD по разным настройкам. "
                               "На каждой эпохе учитываются доступные запуски; итоговые метрики берутся с их итоговых эпох. "
                               "Это описательный разброс, включая различия протоколов; ДИ не вычисляется."
                               if mixed else "ДИ среднего по независимым начальным значениям при фиксированной выборке; "
                               "не является погрешностью для генеральной совокупности. При одном запуске разброс не определён.")))
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    pd.DataFrame([{k: v for k, v in row.items() if k != "history"} for row in rows]).to_csv(output / "per_seed.csv", index=False)
    _plot_seed_report(config, rows, summary, output, band, show,
                      include_validation=has_validation, vision_style=True, legend_loc=legend_loc,
                      legend_on=legend_on, legend_ncol=legend_ncol, panel_positions=panel_positions)
    return pd.DataFrame(summary)
