"""Training stages, safe checkpoints and held-out evaluation."""

import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import pennylane as qml
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import HybridClassifier, ModelConfig
from .data import audit_splits, class_distribution_table


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(x, y, batch_size, shuffle=False, seed=42):
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                      batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed))


def predict_logits(model, x, batch_size=64):
    model.eval()
    with torch.no_grad():
        batches = [model(torch.from_numpy(x[start:start + batch_size]))
                   for start in range(0, len(x), batch_size)]
    if not batches:
        raise ValueError("Cannot predict an empty dataset")
    return torch.cat(batches)


def evaluate(model, x, y, classes, batch_size=64):
    logits = predict_logits(model, x, batch_size)
    return classification_metrics(logits, y, classes)


def classification_metrics(logits, y, classes):
    prediction = logits.argmax(dim=-1).numpy()
    labels = list(range(len(classes)))
    return dict(
        loss=float(nn.functional.cross_entropy(logits, torch.from_numpy(y))),
        accuracy=float(accuracy_score(y, prediction)),
        balanced_accuracy=float(balanced_accuracy_score(y, prediction)),
        macro_f1=float(f1_score(y, prediction, labels=labels, average="macro", zero_division=0)),
        confusion_matrix=confusion_matrix(y, prediction, labels=labels).tolist(),
        classification_report=classification_report(y, prediction, labels=labels,
                                                      target_names=classes, output_dict=True, zero_division=0),
        n_samples=len(y),
    )


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def save_checkpoint(path, model, metadata, history, epoch, run_config):
    # Only tensors and primitive Python types: load with weights_only=True, no arbitrary pickle code.
    torch.save(dict(format_version=1, model_config=model.config_dict(), mode=model.mode_dict(),
                    state_dict=model.state_dict(), data=metadata, history=history,
                    epoch=epoch, run_config=run_config,
                    versions=dict(torch=str(torch.__version__), pennylane=str(qml.__version__),
                                  numpy=str(np.__version__))), path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    model = HybridClassifier(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.set_mode(**checkpoint["mode"])
    return model, checkpoint


def fit(model, data, output, *, epochs=20, batch_size=64, lr=1e-3,
        quantum_lr=None, weight_decay=0.0, patience=7, seed=42, run_config=None,
        export_test_predictions=False):
    if epochs < 1 or batch_size < 1 or patience < 0:
        raise ValueError("epochs and batch_size must be positive; patience must be nonnegative")
    if not 0 < lr < float("inf") or (quantum_lr is not None and not 0 < quantum_lr < float("inf")):
        raise ValueError("Learning rates must be positive and finite")
    if not 0 <= weight_decay < float("inf"):
        raise ValueError("weight_decay must be nonnegative and finite")
    output = Path(output)
    if any((output / filename).exists() for filename in ("best.pt", "last.pt", "metrics.json")):
        raise ValueError(f"Output directory already contains a run: {output}. Choose a new --output directory.")
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    train_x, train_y = data.split("train")
    val_x, val_y = data.split("val")
    classes = data.metadata["classes"]
    split_audit = audit_splits(data.y, classes, data.metadata["indices"])
    write_json(output / "split_audit.json", split_audit)
    class_distribution_table(data).to_csv(output / "class_distribution.csv", index=False)
    loader = make_loader(train_x, train_y, batch_size, shuffle=True, seed=seed)
    classical_params = [p for module in (model.encoder, model.head)
                        for p in module.parameters() if p.requires_grad]
    quantum_params = [p for p in model.quantum.parameters() if p.requires_grad]
    groups = []
    if classical_params:
        groups.append(dict(params=classical_params, lr=lr))
    if quantum_params:
        groups.append(dict(params=quantum_params, lr=quantum_lr if quantum_lr is not None else lr))
    optimizer = torch.optim.Adam(groups, weight_decay=weight_decay) if groups else None
    run_config = dict(run_config or {}, seed=seed, epochs=epochs, batch_size=batch_size,
                      lr=lr, quantum_lr=quantum_lr, weight_decay=weight_decay, patience=patience)
    initial_val = evaluate(model, val_x, val_y, classes, batch_size)
    print(f"Initial val: loss={initial_val['loss']:.4f}, accuracy={initial_val['accuracy']:.4f}", flush=True)
    history = []
    best_loss = initial_val["loss"]
    best_epoch = 0
    save_checkpoint(output / "best.pt", model, data.metadata, history, 0, run_config)
    start = time.perf_counter()
    stale = 0
    if optimizer is not None:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            correct = 0
            for features, target in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(features)
                loss = nn.functional.cross_entropy(logits, target)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss; check input data and learning rate")
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(target)
                correct += int((logits.detach().argmax(-1) == target).sum())
            val = evaluate(model, val_x, val_y, classes, batch_size)
            row = dict(epoch=epoch, train_loss=total_loss / len(train_y),
                       train_accuracy=correct / len(train_y), val_loss=val["loss"],
                       val_accuracy=val["accuracy"], val_macro_f1=val["macro_f1"],
                       elapsed_seconds=time.perf_counter() - start)
            history.append(row)
            print(f"Epoch {epoch:3d}/{epochs}: train_loss={row['train_loss']:.4f} "
                  f"val_loss={val['loss']:.4f} val_acc={val['accuracy']:.4f} "
                  f"val_macro_f1={val['macro_f1']:.4f}", flush=True)
            if val["loss"] < best_loss:
                best_loss, best_epoch, stale = val["loss"], epoch, 0
                save_checkpoint(output / "best.pt", model, data.metadata, history, epoch, run_config)
            else:
                stale += 1
            save_checkpoint(output / "last.pt", model, data.metadata, history, epoch, run_config)
            write_json(output / "history.json", history)
            if patience > 0 and stale >= patience:
                print(f"Early stopping; best epoch: {best_epoch}", flush=True)
                break
    else:
        print("All active parameters are frozen: evaluating without optimization.", flush=True)
        save_checkpoint(output / "last.pt", model, data.metadata, history, 0, run_config)
    best_model, _ = load_checkpoint(output / "best.pt")
    test_x, test_y = data.split("test")
    test_logits = predict_logits(best_model, test_x, batch_size)
    if export_test_predictions:
        np.savez_compressed(output / "test_predictions.npz",
                            indices=np.asarray(data.metadata["indices"]["test"], dtype=np.int64),
                            y_true=test_y, probabilities=test_logits.softmax(-1).numpy())
    metrics = dict(
        best_epoch=best_epoch, elapsed_seconds=time.perf_counter() - start,
        mode=model.mode_dict(), model_config=model.config_dict(),
        n_trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        splits={name: len(values) for name, values in data.metadata["indices"].items()},
        duplicates_dropped=data.metadata["duplicates_dropped"], classes=classes,
        split_audit=split_audit,
        initial_validation=initial_val,
        validation=evaluate(best_model, val_x, val_y, classes, batch_size),
        test=classification_metrics(test_logits, test_y, classes),
        run_config=run_config,
    )
    write_json(output / "history.json", history)
    write_json(output / "metrics.json", metrics)
    write_json(output / "data_metadata.json", data.metadata)
    if history:
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    print(f"Test (best val checkpoint): accuracy={metrics['test']['accuracy']:.4f}, "
          f"macro_f1={metrics['test']['macro_f1']:.4f}; saved to {output}", flush=True)
    return metrics
