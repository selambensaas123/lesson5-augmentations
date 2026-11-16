"""Run augmentation, dataset, benchmarking and transfer-learning experiments."""
import json
import os
from pathlib import Path
import random
from time import perf_counter
import tracemalloc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms as T

from augmentations_basics.datasets import CustomImageDataset
from augmentations_basics.extra_augs import AddGaussianNoise, CutOut, AutoContrast
from homework_augmentations import AugmentationPipeline, blur, shear, brightness_contrast

RESULTS = Path('results')


def save(fig, name):
    """Write a reproducible plot."""
    fig.tight_layout()
    fig.savefig(RESULTS / (name + '.png'), dpi=110)
    plt.close(fig)


def split_indices(labels):
    """Reserve a seeded validation subset independently within each class."""
    rng = random.Random(42)
    train, val = [], []
    for label in sorted(set(labels)):
        indices = [i for i, value in enumerate(labels) if value == label]
        if len(indices) < 2:
            raise ValueError('Every class needs at least two images')
        rng.shuffle(indices)
        count = max(1, round(.2 * len(indices)))
        val.extend(indices[:count])
        train.extend(indices[count:])
    return train, val


def standard(size, force=False):
    """Define the five standard transforms and probabilities."""
    return dict(flip=T.RandomHorizontalFlip(1. if force else .5),
                crop=T.RandomCrop(size, padding=max(2, size // 12)),
                jitter=T.ColorJitter(.35, .35, .25),
                rotation=T.RandomRotation(22),
                gray=T.RandomGrayscale(1. if force else .15))


def pipelines(size):
    """Construct light, medium and heavy ordered pipelines."""
    operations = standard(size)
    result = {}
    for name, selected in [('light', ('flip',)), ('medium', ('flip', 'crop', 'jitter')),
                           ('heavy', tuple(operations))]:
        pipeline = AugmentationPipeline()
        for key in selected:
            pipeline.add_augmentation(key, operations[key])
        if name == 'heavy':
            pipeline.add_augmentation('blur', blur)
        result[name] = pipeline
    return result


def gallery(photos, operations, name):
    """Apply each transform to the same five inputs and render thumbnails."""
    fig, axes = plt.subplots(len(photos), len(operations), figsize=(2 * len(operations), 9))
    for i, photo in enumerate(photos):
        for j, (title, operation) in enumerate(operations.items()):
            axis = axes[i, j]
            axis.imshow(operation(photo.copy()))
            axis.axis('off')
            if i == 0:
                axis.set_title(title)
    save(fig, name)


def inspect(dataset, train):
    """Compute raw sizes and illustrate standard, custom and configured variants."""
    sizes = []
    for path in dataset.images:
        with Image.open(path) as image:
            sizes.append(image.size)
    sizes = np.asarray(sizes)
    counts = np.bincount(dataset.labels, minlength=len(dataset.classes))
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.bar(range(len(counts)), counts)
    left.set(xlabel='Class ID', ylabel='Images')
    right.scatter(sizes[:, 0], sizes[:, 1], alpha=.45)
    right.set(xlabel='Width', ylabel='Height')
    save(fig, 'dataset')
    choices = [next(i for i in train if dataset.labels[i] == cls)
               for cls in range(len(counts))][:5]
    images = [dataset[i][0] for i in choices]
    original = {'original': lambda x: x}
    operations = standard(128, force=True)
    gallery(images, {**original, **operations, 'combined': T.Compose(list(operations.values()))},
            'standard')
    def pil_transform(operation):
        return lambda image: T.ToPILImage()(operation(T.ToTensor()(image)).clamp(0, 1))
    gallery(images, {**original, 'blur': blur, 'shear': shear, 'bright/contrast': brightness_contrast,
                     'noise': pil_transform(AddGaussianNoise()),
                     'cutout': pil_transform(CutOut(p=1)),
                     'autocontrast': pil_transform(AutoContrast(p=1))}, 'custom')
    configs = pipelines(128)
    gallery(images, {**original, **{k: v.apply for k, v in configs.items()}}, 'pipelines')
    for index in train:
        image, label = dataset[index]
        for name, pipeline in configs.items():
            folder = (RESULTS / 'generated') / name / str(label)
            folder.mkdir(parents=True, exist_ok=True)
            pipeline.apply(image).save(folder / f'{index}.jpg')
    return dict(classes=dataset.classes, counts=counts.tolist(), widths_heights={
        'min': sizes.min(0).tolist(), 'max': sizes.max(0).tolist(), 'mean': sizes.mean(0).tolist()})


def benchmark(root, dataset, train):
    """Time 100 distinct loads and report peak Python-tracked allocations."""
    if len(train) < 100:
        raise ValueError('Need 100 distinct training images for the benchmark')
    output = []
    for size in (64, 128, 224, 512):
        resized = CustomImageDataset(root, target_size=(size, size))
        resized.images, resized.labels = dataset.images, dataset.labels
        # Preserve the indices even if directory enumeration order changes.
        pipeline = pipelines(size)['heavy']
        load_time = transform_time = 0.
        tracemalloc.start()
        for i in train[:100]:
            start = perf_counter()
            image = resized[i][0]
            middle = perf_counter()
            result = pipeline.apply(image)
            end = perf_counter()
            load_time += middle - start
            transform_time += end - middle
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        output.append(dict(size=size, load_ms=load_time * 1000,
                           transform_ms=transform_time * 1000, python_peak_bytes=peak,
                           rgb_bytes=2 * 3 * size ** 2))
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    x = [row['size'] for row in output]
    for key in ('load_ms', 'transform_ms'):
        left.plot(x, [row[key] for row in output], marker='o', label=key)
    for key in ('python_peak_bytes', 'rgb_bytes'):
        right.plot(x, [row[key] / 1024 for row in output], marker='o', label=key)
    left.set(ylabel='Milliseconds / 100'); right.set(ylabel='KiB')
    for axis in (left, right):
        axis.set(xlabel='Image size'); axis.legend()
    save(fig, 'benchmark')
    return output


def training_epoch(model, loader, optimizer=None):
    """Compute accuracy and mean cross-entropy for a split."""
    model.train(optimizer is not None)
    total = correct = count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            count += len(y)
    return dict(loss=total / count, accuracy=correct / count, n=count)


def fine_tune(root, dataset, train, val, classes):
    """Fine-tune pretrained MobileNetV3 Small and evaluate held-out images."""
    norm = T.Normalize((.485, .456, .406), (.229, .224, .225))
    aug = T.Compose([pipelines(128)['medium'].apply, T.ToTensor(), norm])
    plain = T.Compose([T.ToTensor(), norm])
    training = CustomImageDataset(root, transform=aug, target_size=(128, 128))
    validation = CustomImageDataset(root, transform=plain, target_size=(128, 128))
    for split in (training, validation):
        split.images, split.labels = dataset.images, dataset.labels
    local = Path('data/weights/mobilenet_v3_small-047dcff4.pth')
    if local.exists():
        import hashlib
        with local.open('rb') as file:
            if not hashlib.file_digest(file, 'sha256').hexdigest().startswith('047dcff4'):
                raise ValueError('MobileNet checkpoint hash mismatch')
        model = models.mobilenet_v3_small()
        model.load_state_dict(torch.load(local, weights_only=True))
    else:
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
    train_loader = DataLoader(Subset(training, train), batch_size=16, shuffle=True)
    val_loader = DataLoader(Subset(validation, val), batch_size=16)
    history = []
    for epoch in range(1, 4):
        row = dict(epoch=epoch, train=training_epoch(model, train_loader, optimizer),
                   val=training_epoch(model, val_loader))
        history.append(row)
        print(row, flush=True)
    torch.save(dict(state=model.state_dict(), classes=classes), RESULTS / 'mobilenet.pt')
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for axis, metric in zip(axes, ('loss', 'accuracy')):
        for split in ('train', 'val'):
            axis.plot([r['epoch'] for r in history], [r[split][metric] for r in history],
                      marker='o', label=split)
        axis.set(xlabel='Epoch', ylabel=metric, xticks=[1, 2, 3])
        axis.legend()
    save(fig, 'training')
    return dict(history=history)


def main():
    """Execute all six tasks and save a machine-readable result."""
    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    RESULTS.mkdir(exist_ok=True)
    root = Path(os.environ.get('LESSON5_TRAIN', 'data/train'))
    dataset = CustomImageDataset(root, target_size=(128, 128))
    import hashlib
    pairs = sorted(zip(dataset.images, dataset.labels))
    seen = set()
    unique = []
    for path, label in pairs:
        with open(path, 'rb') as image:
            digest = hashlib.file_digest(image, 'sha256').digest()
        if digest not in seen:
            seen.add(digest)
            unique.append((path, label))
    dataset.images, dataset.labels = map(list, zip(*unique))
    train, val = split_indices(dataset.labels)
    data = inspect(dataset, train)
    results = dict(dataset=data, benchmark=benchmark(root, dataset, train),
                   training=fine_tune(root, dataset, train, val, dataset.classes))
    (RESULTS / 'metrics.json').write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                         encoding='utf-8')


if __name__ == '__main__':
    main()
