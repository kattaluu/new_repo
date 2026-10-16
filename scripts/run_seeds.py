"""Run exported notebook study configurations or collect separate completed runs."""

import argparse
import json
import os
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache" / "matplotlib"))
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Append independently initialized seeds to a study")
    run.add_argument("--config", required=True, help="JSON exported by the notebook")
    run.add_argument("--seeds", required=True, help="Comma-separated seeds, e.g. 11,22,33,44,55")
    run.add_argument("--output", required=True, help="Study directory, can be reused for other seeds")
    run.add_argument("--skip-existing", action="store_true")
    run.add_argument("--data", help="Override source path after moving an identical dataset")
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--band", choices=["sd", "ci95"], default="sd")
    run.add_argument("--folds", type=int, help="Override config: 0=holdout, k>=2=stratified k-fold")
    run.add_argument("--cv-val-size", type=float, help="Validation fraction of the outer training pool")
    collect = commands.add_parser("collect", help="Pool compatible seeds from one or several study/seed directories")
    collect.add_argument("--inputs", nargs="+", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--band", choices=["sd", "ci95"], default="sd")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import torch
    from qml_beans.seeds import make_study_config, report_seed_study, run_seed_study

    try:
        if args.command == "run":
            if args.threads < 1:
                raise ValueError("threads must be positive")
            torch.set_num_threads(args.threads)
            config = json.loads(Path(args.config).read_text(encoding="utf-8"))
            if args.data:
                config["source"] = str(Path(args.data).resolve())
            if args.folds is not None or args.cv_val_size is not None:
                folds = args.folds if args.folds is not None else config.get("cv_folds", 0)
                if args.cv_val_size is not None and not folds:
                    raise ValueError("--cv-val-size requires k-fold mode")
                config = make_study_config(config["source"], config["model_config"], config["stages"],
                                            target=config["target"], deduplicate=config["deduplicate"],
                                            split_policy=config["split_policy"], split_seed=config["split_seed"],
                                            val_size=args.cv_val_size if args.cv_val_size is not None else config["val_size"],
                                            test_size=config["test_size"], cv_folds=folds)
            if args.band == "ci95" and config["split_policy"] != "fixed":
                raise ValueError("Use --band sd for varying splits")
            study = run_seed_study(config, args.seeds, args.output, skip_existing=args.skip_existing)
            result = report_seed_study([study], study / "report", band=args.band, show=False)
        else:
            result = report_seed_study(args.inputs, args.output, band=args.band, show=False)
        print("\n".join(result["statements"]))
        print("Report:", result["output"])
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
