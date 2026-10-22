"""Official image splits, stratified validation, native IDX/CIFAR readers (no torchvision)."""

from dataclasses import dataclass
import gzip
import hashlib
from pathlib import Path
import struct
import tarfile
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.nn.functional import interpolate


DATASET_CLASSES = {
    "MNIST": tuple(str(i) for i in range(10)),
    "Fashion MNIST": ("Футболка", "Брюки", "Свитер", "Платье", "Пальто", "Сандалии",
                      "Рубашка", "Кроссовки", "Сумка", "Ботинки"),
    "CIFAR-10": ("Самолёт", "Автомобиль", "Птица", "Кошка", "Олень", "Собака",
                 "Лягушка", "Лошадь", "Корабль", "Грузовик"),
}
_FILES = ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
          "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
_MD5 = {
    "MNIST": ("f68b3c2dcbeaaa9fbdd348bbdeb94873", "d53e105ee54ea40749a09fcbcd1e9432",
              "9fb629c4189551a2d022fa330f9573f3", "ec29112dd5afa0611ce80d1b7f02629c"),
    "Fashion MNIST": ("8d4fb7e6c68d591d4c3dfef9ec88bf0d", "25c81989df183df01b3e8a0aad5dffbe",
                      "bef4ecab320f06d8554ea6380940ec79", "bb300cfdad3c16e7a12a480ee83cd310"),
}


def _download(url, path, md5, download):
    def valid(candidate):
        if not candidate.is_file():
            return False
        with candidate.open("rb") as stream:
            return hashlib.file_digest(stream, "md5").hexdigest() == md5

    if valid(path):
        return path
    if not download:
        raise FileNotFoundError(f"Файл отсутствует или повреждён: {path}. Включите download=True.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    urls = (url,) if isinstance(url, str) else tuple(url)
    try:
        for index, address in enumerate(urls):
            try:
                print(f"Загрузка {address}", flush=True)
                with urllib.request.urlopen(address, timeout=30) as source, temporary.open("wb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
                if not valid(temporary):
                    raise ValueError(f"Контрольная сумма не совпадает: {path.name}")
                temporary.replace(path)
                break
            except (OSError, urllib.error.URLError) as error:
                if index + 1 == len(urls):
                    raise
                print(f"Источник недоступен: {error}. Пробуем зеркало.", flush=True)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_idx(path):
    with gzip.open(path, "rb") as stream:
        raw = stream.read()
    if raw[:3] != b"\x00\x00\x08" or raw[3] not in (1, 3):
        raise ValueError("Ожидался IDX с байтовыми метками или изображениями")
    ndim = raw[3]
    shape = struct.unpack(">" + "I" * ndim, raw[4:4 + 4 * ndim])
    data = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim)
    if data.size != np.prod(shape):
        raise ValueError("Некорректная длина IDX")
    return data.reshape(shape).copy()


def load_raw_images(name, root, *, download=True):
    if name not in DATASET_CLASSES:
        raise ValueError("Выберите MNIST, Fashion MNIST или CIFAR-10")
    folder = Path(root) / name.replace(" ", "_")
    if name != "CIFAR-10":
        base = ("https://storage.googleapis.com/cvdf-datasets/mnist/" if name == "MNIST"
                else "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/")
        arrays = [read_idx(_download(base + filename, folder / filename, md5, download))
                  for filename, md5 in zip(_FILES, _MD5[name])]
        return arrays[0][:, None], arrays[1], arrays[2][:, None], arrays[3]
    archive = _download(("https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-binary.tar.gz",
                         "https://mirror.tensorflow.org/www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz",
                         "https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz",
                         "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"),
                        folder / "cifar-10-binary.tar.gz", "c32a1d4ab5d03f1284b67883e8d87530", download)
    with tarfile.open(archive, "r:gz") as stream:
        def read(name):
            # Read only explicit members, without extracting paths or deserializing pickle.
            with stream.extractfile("cifar-10-batches-bin/" + name) as member:
                records = np.frombuffer(member.read(), dtype=np.uint8).reshape(-1, 3073)
            return records[:, 1:].reshape(-1, 3, 32, 32).copy(), records[:, 0].copy()
        parts = [read(f"data_batch_{index}.bin") for index in range(1, 6)]
        test_x, test_y = read("test_batch.bin")
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), test_x, test_y


@dataclass
class VisionData:
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    classes: tuple[str, ...]
    metadata: dict

    def distribution(self):
        return pd.DataFrame([
            dict(Выборка=label, Класс=name, Количество=int((y == index).sum()),
                 Доля=float((y == index).float().mean()))
            for label, y in (("Обучающая", self.train_y), ("Валидационная", self.val_y), ("Тестовая", self.test_y))
            if len(y)
            for index, name in enumerate(self.classes)
        ])


def image_features(images, image_size, *, grayscale=False):
    """NCHW uint8 -> flattened float32 in [-1, 1]; grayscale before resize."""
    x = torch.from_numpy(np.asarray(images).copy()).float() / 255
    if grayscale:
        if x.shape[1] == 3:
            weights = x.new_tensor([.299, .587, .114]).reshape(1, 3, 1, 1)
            x = (x * weights).sum(dim=1, keepdim=True)
        elif x.shape[1] != 1:
            raise ValueError("Grayscale: ожидается 1 или 3 канала")
    if len(x) == 0:
        return x.new_empty((0, x.shape[1] * image_size * image_size))
    x = interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False, antialias=True)
    return (2 * x - 1).flatten(1)


def prepare_vision_arrays(train_x, train_y, test_x, test_y, *, name, image_size=14,
                          val_size=0.15, seed=42, train_limit=0, val_limit=0, test_limit=0, grayscale=False):
    if type(grayscale) is not bool:
        raise ValueError("grayscale: True или False")
    if name not in DATASET_CLASSES or not 0 <= val_size < 1 or image_size < 2:
        raise ValueError(f"Некорректные настройки данных: dataset={name!r}, image_size={image_size!r}, "
                         f"val_size={val_size!r}. Допустимо: известный датасет, сторона >= 2, "
                         "0 <= val_size < 1 (0 — без валидации).")
    if any(type(limit) is not int or limit < 0 for limit in (train_limit, val_limit, test_limit)):
        raise ValueError("Лимиты должны быть неотрицательными целыми (0 = без ограничения)")
    train_y, test_y = np.asarray(train_y, dtype=np.int64), np.asarray(test_y, dtype=np.int64)
    if not (set(train_y) == set(test_y) == set(range(10))):
        raise ValueError("Нужны все 10 классов с метками 0..9 в официальных выборках")
    if val_size:
        train_indices, val_indices = train_test_split(np.arange(len(train_y)), test_size=val_size,
                                                     random_state=seed, stratify=train_y)
    else:
        train_indices, val_indices = np.arange(len(train_y)), np.empty(0, dtype=np.int64)

    def subset(indices, y, limit):
        if limit and limit < len(indices):
            indices, _ = train_test_split(indices, train_size=limit, random_state=seed, stratify=y[indices])
        return np.sort(indices)

    train_indices = subset(train_indices, train_y, train_limit)
    val_indices = subset(val_indices, train_y, val_limit)
    test_indices = subset(np.arange(len(test_y)), test_y, test_limit)
    channels = 1 if grayscale else int(train_x.shape[1])

    def features(images, indices):
        if not len(indices):
            return torch.empty((0, channels * image_size * image_size), dtype=torch.float32)
        chunks = []
        for offset in range(0, len(indices), 512):
            chunks.append(image_features(images[indices[offset:offset + 512]], image_size, grayscale=grayscale))
        return torch.cat(chunks)

    metadata = dict(dataset=name, seed=seed, image_size=image_size, channels=channels,
                    normalization="pixel/127.5 - 1; bilinear resize; no fitted statistics",
                    official_train_size=len(train_y), official_test_size=len(test_y), val_size=val_size,
                    subset=bool(train_limit or (val_limit if val_size else 0) or test_limit),
                    indices=dict(train=train_indices.tolist(), val=val_indices.tolist(), test=test_indices.tolist()))
    if grayscale:
        metadata["grayscale"] = True
        metadata["normalization"] = "RGB -> 0.299 R + 0.587 G + 0.114 B; " + metadata["normalization"]
    digest = hashlib.sha256()
    for array in (train_x, train_y, test_x, test_y):
        digest.update(memoryview(np.ascontiguousarray(array)))
    metadata["dataset_sha256"] = digest.hexdigest()
    return VisionData(features(train_x, train_indices), torch.from_numpy(train_y[train_indices]),
                      features(train_x, val_indices), torch.from_numpy(train_y[val_indices]),
                      features(test_x, test_indices), torch.from_numpy(test_y[test_indices]),
                      DATASET_CLASSES[name], metadata)


def load_vision_data(name, root, *, download=True, **kwargs):
    return prepare_vision_arrays(*load_raw_images(name, root, download=download), name=name, **kwargs)
