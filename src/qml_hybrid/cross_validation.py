"""Validated out-of-fold collection: folds are not independent seed replicates."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

from .data import audit_splits
from .model import saved_model_config, ModelConfig
from .training import write_json


def _load_predictions(row):
    with np.load(row["oof_file"], allow_pickle=False) as archive:
        indices = archive["indices"].copy()
        y = archive["y_true"].copy()
        probabilities = archive["probabilities"].astype(np.float64)
    classes = row["metadata"]["classes"]
    if (indices.dtype.kind not in "iu" or y.dtype.kind not in "iu" or y.shape != indices.shape or
            probabilities.shape != (len(indices), len(classes)) or not np.isfinite(probabilities).all() or
            (probabilities < 0).any() or (probabilities > 1 + 1e-6).any() or
            not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6) or
            (y < 0).any() or (y >= len(classes)).any() or
            indices.tolist() != row["metadata"]["indices"]["test"]):
        raise ValueError(f"Invalid out-of-fold predictions in {row['oof_file']}")
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    prediction = probabilities.argmax(axis=1)
    scores = dict(test_accuracy=float(accuracy_score(y, prediction)),
                  test_balanced_accuracy=float(balanced_accuracy_score(y, prediction)),
                  test_macro_f1=float(f1_score(y, prediction, labels=list(range(len(classes))),
                                               average="macro", zero_division=0)))
    if any(abs(row[name] - score) > 1e-10 for name, score in scores.items()):
        raise ValueError(f"OOF predictions disagree with saved fold metrics in {row['oof_file']}")
    return indices, y, probabilities


def combine_oof(rows):
    """Reconstruct one prediction per original dataset row, rejecting overlaps or missing folds."""
    metadata = rows[0]["metadata"]
    n = sum(len(values) for values in metadata["indices"].values())
    probabilities = np.empty((n, len(metadata["classes"])), dtype=np.float64)
    y = np.empty(n, dtype=np.int64)
    seen = np.zeros(n, dtype=np.int64)
    for row in rows:
        indices, true, probability = _load_predictions(row)
        if (indices < 0).any() or (indices >= n).any() or len(set(indices)) != len(indices):
            raise ValueError("OOF indices are duplicated or outside the dataset")
        seen[indices] += 1
        y[indices] = true
        probabilities[indices] = probability
    if not np.all(seen == 1):
        raise ValueError("OOF test folds must predict every object exactly once; missing/overlapping fold")
    for row in rows:
        verified = audit_splits(y, metadata["classes"], row["metadata"]["indices"])
        if verified != row["metadata"]["split_audit"]:
            raise ValueError("Saved class-distribution audit differs from OOF targets")
    return y, probabilities


def read_cv_run(config, saved, record, split_seed):
    """Read all folds of one completed seed, including protocol and coverage checks."""
    folds = saved.get("folds", [])
    if saved["split_seed"] != split_seed or len(folds) != config["cv_folds"]:
        raise ValueError(f"Incomplete/mismatching CV folds in {record}")
    rows = []
    for fold_index, fold in enumerate(folds):
        fold_folder = f"fold_{fold_index + 1:02d}"
        if fold["fold"] != fold_index or fold["folder"] != fold_folder or len(fold["stages"]) != len(config["stages"]):
            raise ValueError(f"Invalid fold manifest in {record}")
        for stage_index, (artifact, stage) in enumerate(zip(fold["stages"], config["stages"])):
            folder = f"{stage_index + 1:02d}_{stage['name']}"
            if artifact != dict(name=stage["name"], folder=folder):
                raise ValueError(f"Invalid CV stage in {record}")
            directory = record.parent / fold_folder / folder
            metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            metadata = json.loads((directory / "data_metadata.json").read_text(encoding="utf-8"))
            if (metrics["run_config"].get("study_id") != saved["study_id"] or
                    metrics["run_config"]["seed"] != saved["seed"] or
                    metrics["run_config"].get("split_seed") != split_seed or
                    metrics["run_config"].get("fold") != fold_index or
                    saved_model_config(metrics["model_config"]) != saved_model_config(config["model_config"]) or
                    metadata["fingerprint"] != config["dataset_fingerprint"] or
                    metadata["seed"] != split_seed or metadata["fold"] != fold_index or
                    metadata["n_splits"] != config["cv_folds"] or
                    metadata["split_method"] != "stratified_kfold" or
                    metadata["val_size"] != config["val_size"] or metadata["test_size"] != config["test_size"] or
                    metadata["target"] != config["target"] or metadata["deduplicate"] != config["deduplicate"] or
                    any(metrics["run_config"].get(name) != stage[name] for name in
                        ("epochs", "batch_size", "lr", "quantum_lr", "weight_decay", "patience")) or
                    any(metrics["mode"][name] != stage[name] for name in metrics["mode"])):
                raise ValueError(f"Incompatible CV artifacts in {directory}")
            if not (directory / "best.pt").is_file():
                raise ValueError(f"Missing best checkpoint: {directory}")
            rows.append(dict(seed=saved["seed"], split_seed=split_seed, fold=fold_index + 1,
                             stage=stage["name"], stage_index=stage_index, best_epoch=metrics["best_epoch"],
                             n_train=metrics["splits"]["train"], n_val=metrics["splits"]["val"],
                             n_test=metrics["splits"]["test"], val_accuracy=metrics["validation"]["accuracy"],
                             **{f"test_{name}": metrics["test"][name]
                                for name in ("accuracy", "balanced_accuracy", "macro_f1", "loss")},
                             output=str(directory), oof_file=str(directory / "test_predictions.npz"),
                             metadata=metadata,
                             history=json.loads((directory / "history.json").read_text(encoding="utf-8"))))
    reference = None
    reference_partitions = None
    for stage in config["stages"]:
        selected = [row for row in rows if row["stage"] == stage["name"]]
        y, _ = combine_oof(selected)
        partitions = [row["metadata"]["indices"] for row in selected]
        if reference is not None and (not np.array_equal(y, reference) or partitions != reference_partitions):
            raise ValueError("CV stages must use identical folds and targets")
        reference, reference_partitions = y, partitions
    return rows


def _seed_history(rows):
    records = [dict(fold=row["fold"], **epoch) for row in rows for epoch in row["history"]]
    if not records:
        return [], []
    frame = pd.DataFrame(records)
    history, counts = [], []
    for epoch, group in frame.groupby("epoch", sort=True):
        history.append(dict(epoch=int(epoch), **{metric: float(group[metric].mean()) for metric in
                           ("train_loss", "val_loss", "train_accuracy", "val_accuracy")}))
        counts.append(dict(epoch=int(epoch), available_folds=len(group)))
    return history, counts


def report_cv_study(config, rows, output, *, band, show):
    from .seeds import _plot_seed_report, sample_statistics, study_id, summarize_seed_runs

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    public = [{key: value for key, value in row.items() if key not in ("history", "metadata")} for row in rows]
    pd.DataFrame(public).to_csv(output / "per_fold.csv", index=False)
    audits = [dict(seed=row["seed"], fold=row["fold"], stage=row["stage"], **part)
              for row in rows for part in row["metadata"]["split_audit"]["rows"]]
    pd.DataFrame(audits).to_csv(output / "class_distribution.csv", index=False)
    oof_rows, fold_stats, availability = [], [], []
    reference_targets = None
    reference_partitions = {}
    for stage_index, stage in enumerate(config["stages"]):
        for seed in sorted({row["seed"] for row in rows}):
            selected = [row for row in rows if row["seed"] == seed and row["stage"] == stage["name"]]
            y, probabilities = combine_oof(selected)
            if reference_targets is not None and not np.array_equal(y, reference_targets):
                raise ValueError("OOF targets differ between seeds despite identical dataset identity")
            reference_targets = y
            partitions = [row["metadata"]["indices"] for row in selected]
            if config["split_policy"] == "fixed":
                if stage["name"] in reference_partitions and partitions != reference_partitions[stage["name"]]:
                    raise ValueError("Fixed-fold study contains different partitions between seeds")
                reference_partitions[stage["name"]] = partitions
            prediction = probabilities.argmax(axis=1)
            n_classes = probabilities.shape[1]
            history, counts = _seed_history(selected)
            availability.extend(dict(seed=seed, stage=stage["name"], **row) for row in counts)
            oof_rows.append(dict(seed=seed, stage=stage["name"], stage_index=stage_index,
                                 split_seed=selected[0]["split_seed"], n_folds=len(selected), n_test=len(y),
                                 test_accuracy=float(accuracy_score(y, prediction)),
                                 test_balanced_accuracy=float(balanced_accuracy_score(y, prediction)),
                                 test_macro_f1=float(f1_score(y, prediction, labels=list(range(n_classes)),
                                                              average="macro", zero_division=0)),
                                 test_loss=float(log_loss(y, probabilities, labels=list(range(n_classes)))),
                                 val_accuracy=float(np.mean([row["val_accuracy"] for row in selected])),
                                 history=history))
            classes = selected[0]["metadata"]["classes"]
            predictions = pd.DataFrame(dict(row_index=np.arange(len(y)),
                                            y_true=[classes[index] for index in y],
                                            y_pred=[classes[index] for index in prediction]))
            for index, label in enumerate(classes):
                predictions[f"probability_{label}"] = probabilities[:, index]
            predictions.to_csv(output / f"oof_seed_{seed}_{stage['name']}.csv", index=False)
            for metric in ("test_accuracy", "test_macro_f1", "test_balanced_accuracy", "test_loss"):
                stats = sample_statistics([row[metric] for row in selected], allow_ci=False)
                fold_stats.append(dict(seed=seed, stage=stage["name"], metric=metric, **stats))
    summary = summarize_seed_runs(config, oof_rows)
    seed_frame = pd.DataFrame([{key: value for key, value in row.items() if key != "history"} for row in oof_rows])
    seed_frame.to_csv(output / "oof_per_seed.csv", index=False)
    seed_frame.to_csv(output / "per_seed.csv", index=False)
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(fold_stats).to_csv(output / "fold_descriptive.csv", index=False)
    pd.DataFrame(availability).to_csv(output / "available_folds_per_epoch.csv", index=False)
    interpretation = (
        "Stratified outer k-fold with inner stratified validation. Each object receives exactly one OOF "
        "test prediction per seed and stage. Preprocessing and model weights are fitted from scratch "
        "within each fold; stages transfer weights only inside that fold. OOF accuracy/F1 are computed "
        "on all OOF predictions, not as an unweighted mean of fold metrics. Folds share training data "
        "and are not independent replicates: fold SD is descriptive only; no fold SEM or CI is computed. "
        "Seed summary n counts independent training seeds, never seeds*k. With fixed folds an approximate "
        "95% t-CI describes the mean OOF score over training randomness conditional on this dataset and "
        "these folds, not population accuracy uncertainty. Varying folds omit SEM/CI. With one seed, "
        "seed SD/SEM/CI are undefined. Curves average available folds within each seed, then average seeds; "
        "availability counts are exported and early-stopped curves are not forward-filled."
    )
    statements = []
    for row in summary:
        if row["metric"] != "test_accuracy":
            continue
        text = f"{row['stage']}: OOF accuracy = {100 * row['mean']:.2f}%; seeds={row['n']}; folds={config['cv_folds']}"
        if row["sd"] is not None:
            text += f"; SD across seeds={100 * row['sd']:.2f} percentage points"
        else:
            text += "; seed SD/SEM/CI undefined (n=1)"
        if row["ci_low"] is not None:
            text += f"; approximate 95% t-CI of seed mean [{100 * row['ci_low']:.2f}%, {100 * row['ci_high']:.2f}%]"
        statements.append(text)
    write_json(output / "summary.json", dict(study_id=study_id(config), config=config, band=band,
                                            interpretation=interpretation, statistics=summary,
                                            fold_statistics=fold_stats))
    (output / "report.txt").write_text("\n".join(statements) + "\n\n" + interpretation + "\n", encoding="utf-8")
    _plot_seed_report(config, oof_rows, summary, output, band, show)
    return dict(output=str(output), config=config, rows=seed_frame, summary=pd.DataFrame(summary),
                fold_rows=pd.DataFrame(public), fold_summary=pd.DataFrame(fold_stats),
                class_distribution=pd.DataFrame(audits), statements=statements)
