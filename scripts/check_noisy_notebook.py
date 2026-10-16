"""Execute the actual notebook on real small stratified image subsets, including Windows spawn."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MNIST", choices=["MNIST", "Fashion MNIST", "CIFAR-10"])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--backend", default="torch_density", choices=["torch_density", "pennylane_mixed", "pennylane_statevector"])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache" / "matplotlib"))
    os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(root / ".cache" / "jupyter-runtime"))
    os.environ.setdefault("IPYTHONDIR", str(root / ".cache" / "ipython"))
    if os.name == "nt":
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    import nbformat
    from nbclient import NotebookClient
    notebook = nbformat.read(root / "notebooks" / "noisy_vision_cpu.ipynb", as_version=4)
    additions = {
        "data-controls": f'''dataset.value = {args.dataset!r}
image_size.value = 4
train_limit.value = 20
val_limit.value = 10
test_limit.value = 20''',
        "model-controls": f'''pre_editor.count.value = 1
pre_editor.rows[0][0].value = 8
post_editor.count.value = 1
post_editor.rows[0][0].value = 8
depth.value = 1
topology.value = "none"
backend.value = {args.backend!r}''',
        "training-controls": f'''workers.value = {args.workers}
microbatch.value = 1
batch_size.value = 10
epochs.value = 1
seeds.value = "101,202"
output_name.value = "noisy_notebook_smoke"''',
    }
    for cell in notebook.cells:
        if cell.id in additions:
            cell.source += "\n\n" + additions[cell.id]
    client = NotebookClient(notebook, timeout=600, kernel_name="qml-beans", resources={"metadata": {"path": str(root)}})
    client.execute()
    destination = root / "runs" / f"noisy_notebook_{args.dataset.replace(' ', '_')}_{args.backend}_checked.ipynb"
    nbformat.write(notebook, destination)
    print(destination)


if __name__ == "__main__":
    main()
