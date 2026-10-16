"""Dataset loading, stable splits and train-only preprocessing."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


UCI_URL = "https://archive.ics.uci.edu/static/public/602/dry%2Bbean%2Bdataset.zip"
SONAR_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".arff":
        records, _ = arff.loadarff(path)
        frame = pd.DataFrame(records)
        for column in frame.select_dtypes(include="object"):
            frame[column] = frame[column].map(
                lambda value: value.decode("utf-8") if isinstance(value, bytes) else value
            )
        return frame
    raise ValueError("Supported input formats: .csv and .arff")


def download_dry_bean(cache_dir: str | Path = "data") -> Path:
    destination = Path(cache_dir) / "Dry_Bean_Dataset.arff"
    if destination.is_file():
        return destination
    request = urllib.request.Request(UCI_URL, headers={"User-Agent": "qml-beans/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        archive_bytes = response.read()
    with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
        members = [name for name in archive.namelist() if name.endswith("Dry_Bean_Dataset.arff")]
        if len(members) != 1:
            raise ValueError("Unexpected UCI archive: cannot find the Dry Bean ARFF file")
        payload = archive.read(members[0])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return destination


def download_sonar(cache_dir: str | Path = "data") -> Path:
    """Cache the official headerless UCI data as a CSV with an explicit target column."""
    destination = Path(cache_dir) / "Sonar.csv"
    if destination.is_file():
        return destination
    request = urllib.request.Request(SONAR_URL, headers={"User-Agent": "qml-beans/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    frame = pd.read_csv(BytesIO(payload), header=None)
    if frame.shape != (208, 61) or set(frame.iloc[:, -1]) != {"M", "R"}:
        raise ValueError("Unexpected UCI Sonar data: expected 208 rows, 60 features and M/R labels")
    frame.columns = [f"Attribute{i}" for i in range(1, 61)] + ["Class"]
    features = frame.iloc[:, :-1].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("UCI Sonar contains non-finite features")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def read_dataset(path, target, deduplicate=True):
    frame = load_table(path)
    if target not in frame:
        raise ValueError(f"Target column {target!r} not found. Columns: {list(frame.columns)}")
    if not frame.columns.is_unique:
        raise ValueError("Column names must be unique")
    if frame[target].isna().any():
        raise ValueError("Target labels must not contain missing values")
    original_size = len(frame)
    feature_names = [column for column in frame if column != target]
    if not feature_names:
        raise ValueError("At least one feature column is required")
    try:
        features = frame[feature_names].apply(pd.to_numeric, errors="raise").astype(np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError("Features must be numeric; encode categorical features first") from exc
    if np.isinf(features.to_numpy()).any():
        raise ValueError("Features contain infinity; use finite values or NaN")
    target_values = frame[target].astype(str)
    frame = features.copy()
    frame[target] = target_values
    # Prevent identical examples leaking between splits. Contradictory labels are not guessed.
    if deduplicate:
        frame = frame.drop_duplicates()
        if frame.duplicated(subset=feature_names).any():
            raise ValueError("Identical feature rows have conflicting labels")
    frame = frame.reset_index(drop=True)
    fingerprint = sha256()
    fingerprint.update(frame.to_json(orient="split", double_precision=15).encode("utf-8"))
    return frame, feature_names, fingerprint.hexdigest(), original_size - len(frame)


def transform_features(frame: pd.DataFrame, preprocessing: dict) -> np.ndarray:
    names = preprocessing["feature_names"]
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    values = frame[names].to_numpy(dtype=np.float64, copy=True)
    if np.isinf(values).any():
        raise ValueError("Features contain infinity")
    values = np.where(np.isnan(values), np.asarray(preprocessing["median"]), values)
    values = (values - np.asarray(preprocessing["mean"])) / np.asarray(preprocessing["scale"])
    result = values.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Preprocessing produced non-finite values")
    return result


@dataclass
class PreparedData:
    x: np.ndarray
    y: np.ndarray
    metadata: dict

    def split(self, name):
        indices = self.metadata["indices"][name]
        return self.x[indices], self.y[indices]


def audit_splits(y, classes, indices):
    """Verify a partition and expose class proportions, including integer-rounding deviations."""
    y = np.asarray(y)
    n = len(y)
    if set(indices) != {"train", "val", "test"}:
        raise ValueError("A partition must contain train, val and test")
    all_indices = []
    parts = {"full": np.arange(n)}
    for name, values in indices.items():
        if len(values) == 0 or any(isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) for index in values):
            raise ValueError(f"{name} indices must be nonempty integers")
        part = np.asarray(values, dtype=np.int64)
        if (part < 0).any() or (part >= n).any():
            raise ValueError(f"{name} indices are outside the dataset")
        parts[name] = part
        all_indices.extend(part.tolist())
    if len(all_indices) != n or len(set(all_indices)) != n:
        raise ValueError("Train/val/test must be disjoint and cover every dataset row exactly once")
    counts = np.bincount(y, minlength=len(classes))
    proportions = counts / n
    rows = []
    checks = {}
    for name in ("full", "train", "val", "test"):
        part = parts[name]
        observed = np.bincount(y[part], minlength=len(classes))
        if (observed == 0).any():
            raise ValueError(f"Every class must be represented in {name}; increase split sizes")
        expected = len(part) * proportions
        checks[name] = bool(np.max(np.abs(observed - expected)) <= 1 + 1e-9)
        for label, count, expected_count, reference in zip(classes, observed, expected, proportions):
            rows.append(dict(split=name, label=str(label), count=int(count), size=int(len(part)),
                             proportion=float(count / len(part)), percent=float(100 * count / len(part)),
                             reference_percent=float(100 * reference), expected_count=float(expected_count),
                             count_deviation=float(count - expected_count),
                             deviation_pp=float(100 * (count / len(part) - reference))))
    return dict(disjoint=True, covers_all=True, all_classes_present=True,
                strategy="preserve_original_proportions", rounding_tolerance_objects=1,
                within_one_object=checks, rows=rows)


def class_distribution_table(data):
    audit = audit_splits(data.y, data.metadata["classes"], data.metadata["indices"])
    return pd.DataFrame(audit["rows"])


def _prepare_partition(frame, feature_names, y, metadata):
    """Fit preprocessing for exactly one partition; never reuse a scaler between CV folds."""
    audit = audit_splits(y, metadata["classes"], metadata["indices"])
    train_values = frame.iloc[metadata["indices"]["train"]][feature_names].to_numpy(dtype=np.float64)
    if np.isnan(train_values).all(axis=0).any():
        raise ValueError("A feature is entirely missing in the training split")
    median = np.nanmedian(train_values, axis=0)
    scaler = StandardScaler().fit(np.where(np.isnan(train_values), median, train_values))
    metadata = dict(metadata, split_audit=audit, preprocessing=dict(
        feature_names=feature_names, median=median.tolist(), mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist()))
    return PreparedData(transform_features(frame, metadata["preprocessing"]), y, metadata)


def prepare_cv_data(path, target="Class", *, n_splits=5, seed=42, val_size=0.15, deduplicate=True):
    """Outer stratified test folds; inner stratified validation is a fraction of the outer pool."""
    if type(n_splits) is not int or n_splits < 2:
        raise ValueError("k-fold requires an integer n_splits >= 2")
    if not 0 < val_size < 1:
        raise ValueError("CV validation fraction must lie between 0 and 1")
    frame, feature_names, fingerprint, dropped = read_dataset(path, target, deduplicate)
    encoder = LabelEncoder().fit(frame[target])
    y = encoder.transform(frame[target]).astype(np.int64)
    if len(encoder.classes_) < 2 or np.bincount(y).min() < n_splits:
        raise ValueError("Every class must contain at least k objects for stratified k-fold")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for fold, (pool, test) in enumerate(splitter.split(np.zeros(len(y)), y)):
        inner_seed = (seed + fold) % (2**32)
        try:
            train, val = train_test_split(pool, test_size=val_size, stratify=y[pool], random_state=inner_seed)
        except ValueError as exc:
            raise ValueError(f"Cannot create validation inside fold {fold + 1}: {exc}") from exc
        metadata = dict(target=target, fingerprint=fingerprint, deduplicate=deduplicate,
                        duplicates_dropped=dropped, seed=seed, inner_seed=inner_seed,
                        val_size=val_size, test_size=1 / n_splits, n_splits=n_splits, fold=fold,
                        split_method="stratified_kfold", validation_fraction_basis="outer_training_pool",
                        classes=encoder.classes_.tolist(),
                        indices={"train": train.tolist(), "val": val.tolist(), "test": test.tolist()})
        folds.append(_prepare_partition(frame, feature_names, y, metadata))
    tests = [index for data in folds for index in data.metadata["indices"]["test"]]
    if len(tests) != len(y) or len(set(tests)) != len(y):
        raise ValueError("CV test folds must cover each row exactly once")
    return folds


def prepare_data(path, target="Class", seed=42, test_size=0.2, val_size=0.15,
                 deduplicate=True, saved_metadata=None):
    if saved_metadata is not None:
        target = saved_metadata["target"]
        deduplicate = saved_metadata["deduplicate"]
    frame, feature_names, fingerprint, dropped = read_dataset(path, target, deduplicate)
    labels = frame[target].to_numpy()
    if saved_metadata is not None:
        if fingerprint != saved_metadata["fingerprint"]:
            raise ValueError("Dataset differs from checkpoint (content, order or schema). Use the original dataset.")
        metadata = saved_metadata
        mapping = {label: i for i, label in enumerate(metadata["classes"])}
        y = np.asarray([mapping[label] for label in labels], dtype=np.int64)
    else:
        if not (0 < test_size < 1 and 0 < val_size < 1 and test_size + val_size < 1):
            raise ValueError("Validation and test fractions must be positive and sum to less than 1")
        label_encoder = LabelEncoder().fit(labels)
        y = label_encoder.transform(labels).astype(np.int64)
        if len(label_encoder.classes_) < 2:
            raise ValueError("Dataset must contain at least two classes")
        indices = np.arange(len(frame))
        try:
            train_val, test = train_test_split(indices, test_size=test_size, stratify=y, random_state=seed)
            train, val = train_test_split(train_val, test_size=val_size / (1 - test_size),
                                          stratify=y[train_val], random_state=seed)
        except ValueError as exc:
            raise ValueError(f"Cannot make stratified train/validation/test splits: {exc}") from exc
        expected_classes = set(np.unique(y))
        if any(set(y[part]) != expected_classes for part in (train, val, test)):
            raise ValueError("Every class must occur in every split; increase the dataset or split sizes")
        metadata = dict(target=target, fingerprint=fingerprint, deduplicate=deduplicate,
                        duplicates_dropped=dropped, seed=seed,
                        val_size=val_size, test_size=test_size,
                        classes=label_encoder.classes_.tolist(),
                        indices={"train": train.tolist(), "val": val.tolist(), "test": test.tolist()})
        return _prepare_partition(frame, feature_names, y, metadata)
    metadata = dict(metadata, split_audit=audit_splits(y, metadata["classes"], metadata["indices"]))
    x = transform_features(frame, metadata["preprocessing"])
    return PreparedData(x, y, metadata)
