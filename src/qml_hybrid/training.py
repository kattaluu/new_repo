import json
import random
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def classification_metrics(logits, y, classes):
    prediction = logits.argmax(dim=-1).numpy()
    labels = list(range(len(classes)))
    return dict(loss=float(nn.functional.cross_entropy(logits, torch.from_numpy(y))), accuracy=float(accuracy_score(y, prediction)), balanced_accuracy=float(balanced_accuracy_score(y, prediction)), macro_f1=float(f1_score(y, prediction, labels=labels, average='macro', zero_division=0)), confusion_matrix=confusion_matrix(y, prediction, labels=labels).tolist(), classification_report=classification_report(y, prediction, labels=labels, target_names=classes, output_dict=True, zero_division=0), n_samples=len(y))

def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
