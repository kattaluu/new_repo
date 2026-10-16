import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from .data import download_dry_bean, download_sonar, load_table, prepare_data, transform_features
from .model import HybridClassifier, ModelConfig
from .training import evaluate, fit, load_checkpoint, predict_logits, seed_everything, write_json


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value):
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def add_dataset_args(parser):
    parser.add_argument("--data", help="Local CSV/ARFF; overrides --dataset")
    parser.add_argument("--dataset", choices=["dry-bean", "sonar"], default="dry-bean")
    parser.add_argument("--cache-dir", default="data")


def parser():
    root = argparse.ArgumentParser(description="8-qubit hybrid tabular classifier (CPU state vector)")
    commands = root.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Download a UCI dataset")
    download.add_argument("--cache-dir", default="data")
    download.add_argument("--dataset", choices=["dry-bean", "sonar"], default="dry-bean")
    train = commands.add_parser("train", help="Train a new model or start a stage from a checkpoint")
    add_dataset_args(train)
    train.add_argument("--target", default="Class", help="Target column for a new CSV/ARFF dataset")
    train.add_argument("--checkpoint", help="Restore model, preprocessing, labels and identical splits")
    train.add_argument("--output", help="New run directory; default: runs/<timestamp>")
    for name in ("quantum", "train-classical", "train-quantum"):
        train.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=None,
                           help="Default: enabled for new models, inherited for checkpoints")
    train.add_argument("--pre-hidden", type=positive_int, default=None)
    train.add_argument("--post-hidden", type=positive_int, default=None)
    train.add_argument("--quantum-depth", type=positive_int, default=None)
    train.add_argument("--epochs", type=positive_int, default=20)
    train.add_argument("--batch-size", type=positive_int, default=64)
    train.add_argument("--lr", type=positive_float, default=1e-3)
    train.add_argument("--quantum-lr", type=positive_float)
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--patience", type=int, default=7, help="Early stopping on validation loss; 0 disables")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--val-size", type=float, default=0.15, help="Fraction of the entire dataset")
    train.add_argument("--keep-duplicates", action="store_true", help="Default: deduplicate before splitting")
    train.add_argument("--threads", type=positive_int, default=1, help="PyTorch CPU threads")
    evaluation = commands.add_parser("evaluate", help="Evaluate a saved model on its held-out split")
    add_dataset_args(evaluation)
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--split", choices=["val", "test"], default="test")
    evaluation.add_argument("--output", help="Optional metrics JSON file")
    evaluation.add_argument("--batch-size", type=positive_int, default=64)
    prediction = commands.add_parser("predict", help="Predict labels for new rows using saved preprocessing")
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--data", required=True, help="CSV/ARFF with the original feature columns")
    prediction.add_argument("--output", required=True, help="Prediction CSV with class probabilities")
    prediction.add_argument("--batch-size", type=positive_int, default=64)
    return root


def dataset_path(args):
    loader = download_sonar if args.dataset == "sonar" else download_dry_bean
    return args.data if args.data else loader(args.cache_dir)


def run_train(args):
    if args.patience < 0 or not 0 <= args.weight_decay < float("inf"):
        raise ValueError("patience and weight-decay must be nonnegative and finite")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    path = dataset_path(args)
    if args.checkpoint:
        model, checkpoint = load_checkpoint(args.checkpoint)
        for option in ("pre_hidden", "post_hidden", "quantum_depth"):
            value = getattr(args, option)
            if value is not None and value != getattr(model.config, option):
                raise ValueError(f"Cannot change {option} when loading a checkpoint")
        data = prepare_data(path, saved_metadata=checkpoint["data"])
        mode = model.mode_dict()
    else:
        data = prepare_data(path, target=args.target, seed=args.seed, test_size=args.test_size,
                            val_size=args.val_size, deduplicate=not args.keep_duplicates)
        model = HybridClassifier(ModelConfig(
            n_features=data.x.shape[1], n_classes=len(data.metadata["classes"]),
            pre_hidden=args.pre_hidden or 32, post_hidden=args.post_hidden or 16,
            quantum_depth=args.quantum_depth or 2))
        mode = model.mode_dict()
    for name in mode:
        if getattr(args, name) is not None:
            mode[name] = getattr(args, name)
    model.set_mode(**mode)
    output = args.output or str(Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    print(f"Mode: {mode}; split sizes: "
          f"{ {key: len(value) for key, value in data.metadata['indices'].items()} }; "
          f"duplicates removed: {data.metadata['duplicates_dropped']}", flush=True)
    fit(model, data, output, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, quantum_lr=args.quantum_lr, weight_decay=args.weight_decay,
        patience=args.patience, seed=args.seed,
        run_config=dict(source_data=str(Path(path).resolve()),
                        parent_checkpoint=str(Path(args.checkpoint).resolve()) if args.checkpoint else None))


def main(argv=None):
    command_parser = parser()
    args = command_parser.parse_args(argv)
    try:
        if args.command == "download":
            loader = download_sonar if args.dataset == "sonar" else download_dry_bean
            print(loader(args.cache_dir))
        elif args.command == "train":
            run_train(args)
        elif args.command == "evaluate":
            torch.set_num_threads(1)
            model, checkpoint = load_checkpoint(args.checkpoint)
            data = prepare_data(dataset_path(args), saved_metadata=checkpoint["data"])
            x, y = data.split(args.split)
            metrics = evaluate(model, x, y, data.metadata["classes"], args.batch_size)
            if args.output:
                write_json(args.output, metrics)
            import json
            print(json.dumps(metrics, indent=2, ensure_ascii=False))
        elif args.command == "predict":
            torch.set_num_threads(1)
            if Path(args.data).resolve() == Path(args.output).resolve():
                raise ValueError("Prediction output must differ from the input file")
            model, checkpoint = load_checkpoint(args.checkpoint)
            x = transform_features(load_table(args.data), checkpoint["data"]["preprocessing"])
            probabilities = torch.softmax(predict_logits(model, x, args.batch_size), dim=-1).numpy()
            classes = checkpoint["data"]["classes"]
            result = pd.DataFrame({"prediction": [classes[i] for i in probabilities.argmax(-1)]})
            for i, label in enumerate(classes):
                result[f"probability_{label}"] = probabilities[:, i]
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(args.output, index=False)
            print(f"Saved {len(result)} predictions to {args.output}")
    except (ValueError, OSError) as exc:
        command_parser.exit(2, f"Error: {exc}\n")
