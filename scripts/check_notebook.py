"""Execute the delivered notebook; custom smoke checks Sonar, controls and seed studies."""

import argparse
from copy import deepcopy
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--custom-smoke", action="store_true", help="Sonar, custom circuit, three seeds, collection and predictions")
    parser.add_argument("--cv-smoke", action="store_true", help="Sonar, 5-fold CV x 3 seeds, staged training and OOF report")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(root / ".cache" / "jupyter-runtime"))
    os.environ.setdefault("IPYTHONDIR", str(root / ".cache" / "ipython"))
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache" / "matplotlib"))
    Path(os.environ["JUPYTER_RUNTIME_DIR"]).mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(root / "notebooks" / "hybrid_qml_explorer.ipynb", as_version=4)
    nbformat.validate(notebook)
    if args.custom_smoke or args.cv_smoke:
        additions = {
            "dataset-controls": 'dataset_choice.value = "Sonar (UCI)"',
            "architecture-controls": '''pre_editor.count.value = 2
pre_editor.rows[0][0].value = 12
pre_editor.rows[0][1].value = "gelu"
pre_editor.rows[1][0].value = 6
pre_editor.rows[1][1].value = "tanh"
post_editor.count.value = 0
embedding.value = "RZ"
assert hadamard.value
rotations.value = "RX,RY,RZ"
topology.value = "custom"
edges.value = "[[0,1],[0,4],[4,7]]"
entangler.value = "CZ"
measurement.value = "Y"
reupload.value = True''',
            "training-controls": 'epochs.value = 1\npreset.value = "1. Классика без схемы"',
            "seeds-controls": '''from datetime import datetime
study_seeds.value = "101,202,303"
study_protocol.value = "staged"
study_classical_epochs.value = 1
study_quantum_epochs.value = 1
study_joint_epochs.value = 1
study_output.value = str(ROOT / "runs" / "notebook_seed_smoke" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
study_band.value = "ci95"
study_run.value = True''',
            "collect-controls": '''collect_paths.value = str(study_folder)
collect_output.value = str(study_folder / "collected")''',
        }
        if args.cv_smoke:
            additions["seeds-controls"] += "\nstudy_cv.value = True\nstudy_k.value = 5"
        for cell in notebook.cells:
            if cell.id in additions:
                cell.source += "\n\n" + additions[cell.id]
        by_id = {cell.id: cell for cell in notebook.cells}
        notebook.cells.append(nbformat.v4.new_code_cell('''first_stage = result
before = {name: value.clone() for name, value in experiment.model.state_dict().items()}
preset.value = "2. Только квантовая"
assert use_quantum.value and train_quantum.value and not train_classical.value'''))
        training = deepcopy(by_id["train-stage"])
        training.id = "smoke-quantum-stage"
        notebook.cells.append(training)
        notebook.cells.append(nbformat.v4.new_code_cell('''for name, value in experiment.model.state_dict().items():
    if not name.startswith("quantum."):
        assert torch.equal(before[name], value)
checkpoint_path.value = str(Path(first_stage["output"]) / "best.pt")'''))
        restore = deepcopy(by_id["restore-checkpoint"])
        restore.id = "smoke-restore"
        notebook.cells.append(restore)
        notebook.cells.append(nbformat.v4.new_code_cell('''assert not experiment.model.use_quantum
assert not use_quantum.value
assert experiment.model.config.pre_layers == (12, 6)
assert experiment.model.config.post_layers == ()
assert experiment.model.config.entangler == "CZ"
prediction_path.value = str(source)'''))
        prediction = deepcopy(by_id["predict-new"])
        prediction.id = "smoke-predict"
        notebook.cells.append(prediction)
        notebook.cells.append(nbformat.v4.new_code_cell('''assert len(predictions) == 208
np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
assert set(seed_report["summary"]["n"]) == {3}
assert set(seed_report["rows"]["seed"]) == {101,202,303}
assert len(collected["rows"]) == 9
assert (study_folder / "report" / "metrics_spread.png").is_file()
print("Sonar controls, seed study, collection, stage transfer, restore and predictions: OK")'''))
        if args.cv_smoke:
            notebook.cells.append(nbformat.v4.new_code_cell('''assert len(seed_report["fold_rows"]) == 45
assert set(seed_report["rows"]["n_test"]) == {208}
assert set(seed_report["rows"]["n_folds"]) == {5}
assert (seed_report["class_distribution"].count_deviation.abs() <= 1 + 1e-9).all()
assert set(seed_report["fold_summary"]["n"]) == {5}
assert seed_report["fold_summary"]["ci_low"].isna().all()
print("5-fold CV x 3 seeds, OOF coverage, stratification, collector and n=seed statistics: OK")'''))
    name = "notebook_cv_checked.ipynb" if args.cv_smoke else ("notebook_custom_checked.ipynb" if args.custom_smoke else "notebook_checked.ipynb")
    destination = root / "runs" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = "Sonar CV x seeds smoke" if args.cv_smoke else ("custom Sonar + multi-seed smoke" if args.custom_smoke else "default Dry Bean notebook")
    print("Executing", mode, flush=True)
    client = NotebookClient(notebook, timeout=600, kernel_name="qml-beans",
                            resources={"metadata": {"path": str(root)}})
    try:
        client.execute()
    finally:
        nbformat.write(notebook, destination)
    errors = [output for cell in notebook.cells if cell.cell_type == "code"
              for output in cell.outputs if output.output_type == "error"]
    assert not errors
    print("Executed without errors:", destination, flush=True)


if __name__ == "__main__":
    main()
