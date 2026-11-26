#!/usr/bin/env python3
"""
File: train_cnn.py
Description:
    Trains a genome-specific splice-site CNN with chromosome holdouts,
    validation-selected checkpoints, and a schema-1 model bundle. Can also
    evaluate a trusted saved bundle on CSV windows without retraining.
    Self-contained: remote training needs only this file, a CSV, and dependencies.
Example:
    python scripts/train_cnn.py --train --features data/c_elegans/features.csv
    python scripts/train_cnn.py --evaluate-bundle artifacts/c_elegans/WBcel235/cnn_v001 \\
        --eval-features data/other_species/features.csv --device cpu
Output:
    Training: a new versioned bundle and sibling run directory.
    Evaluation: per-dataset predictions/reports and an appended summary.csv.

Use `python scripts/train_cnn.py --help` for options.
"""

import argparse
import copy
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import logging
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
import time

import matplotlib

if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')  # Remote runs do not need a display server.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, average_precision_score,
    classification_report, f1_score,
)
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

BASES = ('A', 'C', 'G', 'T')
LABEL_ORDER = ['none', 'donor', 'acceptor']
LABEL_TO_INT = {label: i for i, label in enumerate(LABEL_ORDER)}
RANDOM_SEED = 42


# Configuration and small shared utilities

def parse_args(argv=None):
    """Parses the training/evaluation CLI and rejects invalid numeric settings."""
    parser = argparse.ArgumentParser(description='Standalone genome-specific splice-site CNN')
    parser.add_argument('--features', type=Path, default=Path('data/c_elegans/features.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts'))
    parser.add_argument('--genome-id', default='c_elegans')
    parser.add_argument('--species-name', default='Caenorhabditis elegans')
    parser.add_argument('--assembly-id', default='WBcel235')
    parser.add_argument('--model-version', default='cnn_v001')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--train', action='store_true', help='Train and export; otherwise inspect the CSV.')
    mode.add_argument('--evaluate-bundle', type=Path,
                      help='Skip training; score datasets with this trusted saved bundle.')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--device', choices=['auto', 'cuda', 'mps', 'cpu'], default='auto')
    parser.add_argument('--cpu-threads', type=int, default=1)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--eval-features', type=Path, nargs='+',
                        help='One or more evaluation CSVs (default: --features).')
    parser.add_argument('--eval-output-dir', type=Path,
                        help='Evaluation root (default: <bundle>/evaluations).')
    parser.add_argument('--invalid-sequences', choices=['skip', 'error'], default='skip',
                        help='Skip and log invalid rows, or reject the dataset.')
    parser.add_argument('--no-crop', action='store_true',
                        help='Reject longer windows instead of center-cropping.')
    args = parser.parse_args(argv)
    for name in ('batch_size', 'epochs', 'patience', 'cpu_threads', 'log_every'):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} must be positive.')
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('--learning-rate must be finite and positive.')
    return args


def select_device(request):
    """Selects CUDA > MPS > CPU for auto; explicit unavailable devices fail."""
    if request not in ('auto', 'cpu', 'cuda', 'mps'):
        raise ValueError('Device must be auto, cpu, cuda, or mps.')
    if request == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    if request == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Check the PyTorch image and nvidia-smi.')
    if request == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but unavailable. Use a compatible macOS system and '
                           'MPS-enabled PyTorch build, or choose --device cpu.')
    return torch.device(request)


def seed_training(cpu_threads):
    """Sets the safe CPU thread count and seeds before training tensor operations."""
    torch.set_num_threads(cpu_threads)  # Default 1: higher counts previously caused a native crash.
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


def file_sha256(path):
    """Hashes a file in bounded-size blocks for provenance and integrity checks."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Writes readable JSON"""
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def save_checkpoint(payload, path):
    """Replaces a checkpoint only after its temporary file has been written."""
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def setup_logger(run_dir=None):
    """Logs to stdout and, for training, the new run directory."""
    logger = logging.getLogger('splice_cnn')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    handlers = [logging.StreamHandler(sys.stdout)]
    if run_dir is not None:
        handlers.append(logging.FileHandler(run_dir / 'training.log'))
    for handler in handlers:
        handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        logger.addHandler(handler)
    return logger


def training_paths(args):
    """Checks safe identity components and resolves bundle/run destinations."""
    for name in ('genome_id', 'assembly_id', 'model_version'):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', getattr(args, name)):
            raise ValueError(f'{name} must contain only letters, digits, underscores or hyphens.')
    bundle_dir = args.output_dir.expanduser().resolve() / args.genome_id / args.assembly_id / args.model_version
    return bundle_dir, bundle_dir.with_name(bundle_dir.name + '_run')


# Training data: validate -> inspect -> chromosome holdouts -> encode batches

def validate_dataset(frame):
    """
    Validates the training CSV's sequences, labels, and genomic provenance.

    PARAMETERS
    ----------
    frame : pandas.DataFrame
        sequence, label, chrom, strand, and original 1-based position columns

    RETURNS
    -------
    int
        common even sequence length, at least two bases

    RAISES
    ------
    ValueError
        if required values are missing, labels/bases/lengths/strands are invalid,
        or identical sequences have conflicting labels; coordinates are retained,
        not independently checked against an annotation
    """
    required = {'sequence', 'label', 'chrom', 'strand', 'position'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError('Dataset is empty or has missing sequences, labels, or genomic metadata.')
    if set(frame['label'].unique()) != set(LABEL_ORDER):
        raise ValueError(f'Expected labels {LABEL_ORDER}.')
    lengths = frame['sequence'].str.len().unique()
    if len(lengths) != 1 or lengths[0] < 2 or lengths[0] % 2:
        raise ValueError('Sequences must have one even length of at least two.')
    if not frame['sequence'].str.fullmatch(r'[ACGT]+').all():
        raise ValueError('Sequences must contain uppercase A/C/G/T only.')
    if frame.groupby('sequence')['label'].nunique().gt(1).any():
        raise ValueError('Identical sequences have conflicting labels.')
    if not frame['strand'].isin(['+', '-']).all():
        raise ValueError('Expected + or - strand values.')
    return int(lengths[0])


def load_training_table(path):
    """Reads sequence/provenance columns only, ignoring legacy numeric predictors."""
    if not path.is_file():
        raise FileNotFoundError(f'Features file not found (set --features): {path}')
    columns = {'sequence', 'label', 'chrom', 'strand', 'position',
               'negative_type', 'intron_start', 'intron_end'}
    frame = pd.read_csv(path, usecols=lambda column: column in columns)
    return frame, validate_dataset(frame)


def inspect_dataset(frame, length, run_dir=None):
    """Prints class/motif diagnostics and plots unique-sequence GC distributions."""
    print(f'Loaded {len(frame):,} sites; sequence length = {length}')
    print('Exact duplicate rows:', int(frame.duplicated().sum()))
    print('Repeated sequence-label rows:', int(frame.duplicated(['sequence', 'label']).sum()))
    print(frame.head())
    print(frame['label'].value_counts().reindex(LABEL_ORDER).rename('sites'))
    unique = frame.drop_duplicates(['sequence', 'label']).copy()
    unique['gc_content'] = unique['sequence'].map(lambda s: (s.count('G') + s.count('C')) / len(s))
    motifs = []
    for label, motif, offset in (('donor', 'GT', length // 2), ('acceptor', 'AG', length // 2 - 1)):
        sequences = unique.loc[unique['label'].eq(label), 'sequence']
        matches = sequences.str.slice(offset, offset + 2).eq(motif)
        motifs.append({'class': label, 'motif': motif, 'total': len(sequences),
                       'exact_matches': int(matches.sum()), 'match_percentage': 100 * matches.mean()})
    print(pd.DataFrame(motifs))
    fig, ax = plt.subplots(figsize=(8, 4))
    for label in LABEL_ORDER:
        values = unique.loc[unique['label'].eq(label), 'gc_content']
        ax.hist(values, bins=np.linspace(0, 1, 31), density=True, alpha=0.45, label=label)
    ax.set(xlabel='GC fraction', ylabel='Density', title='Unique sequences: GC by class')
    ax.legend()
    fig.tight_layout()
    if run_dir is not None:
        fig.savefig(run_dir / 'gc_by_class.png', dpi=150)
    if 'ipykernel' in sys.modules:
        plt.show()
    plt.close(fig)
    if 'negative_type' in frame.columns:
        print(frame.loc[frame['label'].eq('none'), 'negative_type'].value_counts(dropna=False))


def build_chromosome_folds(frame, n_splits=5):
    """Assigns whole chromosomes largest-first to the currently smallest fold."""
    sizes = frame.groupby('chrom').size().sort_values(ascending=False)
    if len(sizes) < 3:
        raise ValueError('Need at least three chromosomes/contigs for this split.')
    count = min(n_splits, len(sizes))
    if count < 3:
        raise ValueError('Need at least three chromosome folds.')
    folds, totals = [[] for _ in range(count)], [0] * count
    for chrom, size in sizes.items():
        fold = int(np.argmin(totals))
        folds[fold].append(chrom)
        totals[fold] += int(size)
    return folds


def split_by_chromosome(frame, test_chroms, val_chroms):
    """
    Holds out whole chromosomes and removes cross-split exact-sequence overlap.

    Test sequences take precedence over validation, and validation over training.
    Duplicates within a split remain. No random row splitting is performed.

    PARAMETERS
    ----------
    frame : pandas.DataFrame
        validated training table
    test_chroms, val_chroms : iterable[str]
        nonempty, disjoint chromosome groups for test and validation

    RETURNS
    -------
    dict[str, pandas.DataFrame]
        train, validation, and test partitions with reset indices

    RAISES
    ------
    ValueError
        if holdouts overlap, contain unknown chromosomes, or any resulting split
        is empty or missing one of the three classes
    """
    test_chroms, val_chroms = set(test_chroms), set(val_chroms)
    if not test_chroms or not val_chroms or test_chroms & val_chroms:
        raise ValueError('Choose nonempty, disjoint test and validation chromosomes.')
    unknown = (test_chroms | val_chroms) - set(frame['chrom'])
    if unknown:
        raise ValueError(f'Unknown chromosomes: {sorted(unknown)}')
    test = frame.loc[frame['chrom'].isin(test_chroms)].copy()
    val = frame.loc[frame['chrom'].isin(val_chroms)].copy()
    train = frame.loc[~frame['chrom'].isin(test_chroms | val_chroms)].copy()
    before_train, before_val = len(train), len(val)
    val = val.loc[~val['sequence'].isin(test['sequence'])].copy()
    train = train.loc[~train['sequence'].isin(test['sequence'])
                      & ~train['sequence'].isin(val['sequence'])].copy()
    print(f'Removed overlaps: train={before_train - len(train)}, validation={before_val - len(val)}')
    splits = {'train': train, 'validation': val, 'test': test}
    for name, partition in splits.items():
        missing = set(LABEL_ORDER) - set(partition['label'])
        if partition.empty or missing:
            raise ValueError(f'{name} is empty or missing labels {sorted(missing)}; '
                             'choose different chromosome groups.')
    for left, right in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        assert not set(splits[left]['chrom']) & set(splits[right]['chrom'])
        assert not set(splits[left]['sequence']) & set(splits[right]['sequence'])
    return {name: partition.reset_index(drop=True) for name, partition in splits.items()}


def encode_sequences(sequences, expected_length):
    """
    Encodes DNA as contiguous float32 one-hot input with A,C,G,T channels.

    PARAMETERS
    ----------
    sequences : iterable[str]
        nonempty collection of uppercase A/C/G/T sequences
    expected_length : int
        required number of bases per sequence

    RETURNS
    -------
    numpy.ndarray
        shape (number of sequences, 4, expected_length)

    RAISES
    ------
    ValueError
        if the collection is empty, lengths differ, or bases are invalid
    """
    sequences = list(sequences)
    if not sequences or any(len(s) != expected_length for s in sequences):
        raise ValueError('Provide nonempty, equal-length sequences.')
    try:
        bases = np.frombuffer(''.join(sequences).encode('ascii'), dtype='S1')
    except UnicodeEncodeError as error:
        raise ValueError('Sequences must contain uppercase A/C/G/T only.') from error
    bases = bases.reshape(len(sequences), expected_length)
    channels = np.asarray(BASES, dtype='S1')
    encoded = (bases[:, None, :] == channels[None, :, None]).astype(np.float32)
    if not np.all(encoded.sum(axis=1) == 1):
        raise ValueError('Sequences must contain uppercase A/C/G/T only.')
    return np.ascontiguousarray(encoded)


class SequenceDataset(Dataset):
    """Stores strings and integer targets; one-hot encoding happens per batch."""

    def __init__(self, frame):
        """Copies sequence references and maps labels to none/donor/acceptor IDs."""
        self.sequences = frame['sequence'].to_numpy(copy=True)
        self.targets = frame['label'].map(LABEL_TO_INT).to_numpy(dtype=np.int64, copy=True)

    def __len__(self):
        """Returns the number of examples."""
        return len(self.targets)

    def __getitem__(self, index):
        """Returns one unencoded sequence and its integer target."""
        return self.sequences[index], self.targets[index]


def encode_batch(batch, length):
    """Collates examples into float32 (N,4,L) inputs and int64 (N,) targets."""
    sequences, targets = zip(*batch)
    return (torch.from_numpy(encode_sequences(sequences, length)),
            torch.tensor(targets, dtype=torch.int64))


def make_loaders(splits, length, batch_size):
    """Builds per-batch encoders, shuffling training only with its own seeded generator."""
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    return {
        name: DataLoader(SequenceDataset(frame), batch_size=batch_size,
                         shuffle=name == 'train', num_workers=0,
                         generator=generator if name == 'train' else None,
                         collate_fn=partial(encode_batch, length=length))
        for name, frame in splits.items()
    }


# Model, shared inference, and metrics

def make_cnn(length):
    """
    Builds the splice site CNN
    PARAMETERS
    ----------
    length : int
        input sequence length used to size the flattened hidden layer
    RETURNS
    -------
    torch.nn.Sequential
        accepts (N,4,length) input and returns (N,3) logits in none, donor,
        acceptor order; softmax is applied outside the model
    """
    return nn.Sequential(
        nn.Conv1d(4, 16, kernel_size=7, padding=3), nn.ReLU(),
        nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(),
        nn.Flatten(), nn.Linear(32 * length, 64), nn.ReLU(),
        nn.Dropout(0.2), nn.Linear(64, len(LABEL_ORDER)),
    )


def collect_probabilities(model, loader, device):
    """Collects target IDs and uncalibrated softmax scores with dropout/gradients disabled."""
    model.eval()
    truths, scores = [], []
    with torch.inference_mode():
        for inputs, targets in loader:
            scores.append(model(inputs.to(device)).softmax(dim=1).cpu().numpy())
            truths.append(targets.numpy())
    return np.concatenate(truths), np.concatenate(scores)


def score_sequences(model, sequences, length, batch_size, device):
    """Scores CSV windows with the same encoder used by training batches."""
    model.eval()
    scores = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            encoded = encode_sequences(sequences[start:start + batch_size], length)
            scores.append(model(torch.from_numpy(encoded).to(device)).softmax(dim=1).cpu().numpy())
    return np.concatenate(scores) if scores else np.empty((0, len(LABEL_ORDER)), dtype=np.float32)


def classification_metrics(y_true, scores):
    """
    Computes shared classification metrics for training tests and CSV evaluation.

    PARAMETERS
    ----------
    y_true : array-like
        nonempty one-dimensional class IDs in none, donor, acceptor order
    scores : array-like
        (N,3) normalized softmax scores, not logits or calibrated probabilities

    RETURNS
    -------
    tuple[dict, pandas.DataFrame]
        accuracy, macro F1 over present/all classes, and per-class precision,
        recall, F1, support, average precision; average precision is missing when
        a class has no positives or no negatives

    RAISES
    ------
    ValueError
        if target IDs, score shape, or normalization are invalid
    """
    y_true, scores = np.asarray(y_true), np.asarray(scores)
    class_ids = np.arange(len(LABEL_ORDER))
    if y_true.ndim != 1 or not len(y_true) or not set(y_true) <= set(class_ids):
        raise ValueError('Expected nonempty target IDs in none, donor, acceptor order.')
    if scores.shape != (len(y_true), len(LABEL_ORDER)):
        raise ValueError('Expected one score row per example and three columns.')
    if (not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any()
            or not np.allclose(scores.sum(axis=1), 1, atol=1e-5)):
        raise ValueError('Supply normalized softmax scores, not raw logits.')
    predicted = scores.argmax(axis=1)
    report = classification_report(y_true, predicted, labels=class_ids,
                                   target_names=LABEL_ORDER, output_dict=True, zero_division=0)
    per_class = []
    for i, label in enumerate(LABEL_ORDER):
        positives = y_true == i
        per_class.append({
            'class': label, 'precision': report[label]['precision'],
            'recall': report[label]['recall'], 'f1': report[label]['f1-score'],
            'support': int(report[label]['support']),
            'average_precision': (float(average_precision_score(positives, scores[:, i]))
                                  if 0 < positives.sum() < len(y_true) else None),
        })
    summary = {
        'accuracy': float(accuracy_score(y_true, predicted)),
        'macro_f1_present': float(f1_score(y_true, predicted, labels=sorted(set(y_true)),
                                          average='macro', zero_division=0)),
        'macro_f1_all': float(f1_score(y_true, predicted, labels=class_ids, average='macro', zero_division=0)),
    }
    return summary, pd.DataFrame(per_class)


def plot_confusion(y_true, scores, title, path):
    """Saves a true-class-normalized confusion matrix in the fixed label order."""
    ConfusionMatrixDisplay.from_predictions(
        y_true, scores.argmax(axis=1), labels=np.arange(len(LABEL_ORDER)),
        display_labels=LABEL_ORDER, normalize='true', values_format='.2f', cmap='Blues', colorbar=False,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    if 'ipykernel' in sys.modules:
        plt.show()
    plt.close()


# Training and schema-1 bundle export

def train_epoch(model, loader, optimizer, criterion, device, epoch, log_every, logger):
    """Updates weights on training batches and returns the mean cross-entropy loss."""
    model.train()
    started = time.monotonic()
    total_loss, seen = 0.0, 0
    for batch_number, (inputs, targets) in enumerate(loader, 1):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
        seen += len(targets)
        if batch_number == 1 or batch_number % log_every == 0 or batch_number == len(loader):
            elapsed = time.monotonic() - started
            eta = elapsed / batch_number * (len(loader) - batch_number)
            logger.info('Epoch %s batch %s/%s | loss %.4f | %.1f sites/s | train ETA %.0fs',
                        epoch, batch_number, len(loader), total_loss / seen, seen / elapsed, eta)
    return total_loss / seen


def train_cnn(length, train_loader, val_loader, args, device, run_dir, logger):
    """
    Trains with Adam/cross-entropy and restores the validation-selected state.

    Only strictly improved validation macro F1 replaces the best checkpoint.
    Training stops after args.patience stale epochs. Test data is deliberately
    absent from this function. Saved checkpoints do not provide a resume CLI.

    PARAMETERS
    ----------
    length : int
        required input sequence length
    train_loader, val_loader : torch.utils.data.DataLoader
        training and validation partitions, containing all three classes
    args : argparse.Namespace
        epochs, patience, learning_rate, batch_size, and log_every settings
    device : torch.device
        CPU, CUDA, or MPS device on which to train
    run_dir : pathlib.Path
        newly created directory for logs, history, and checkpoints
    logger : logging.Logger
        progress logger

    RETURNS
    -------
    tuple[torch.nn.Module, pandas.DataFrame, float]
        best model in evaluation mode, epoch history, and best validation macro F1

    RAISES
    ------
    RuntimeError
        if no valid checkpoint is produced
    """
    torch.manual_seed(RANDOM_SEED)
    train_loader.generator.manual_seed(RANDOM_SEED)
    model = make_cnn(length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    best_score, best_state, stale_epochs = -1.0, None, 0
    history = []
    logger.info('Starting training: %s sites, %s batches/epoch, batch size %s',
                len(train_loader.dataset), len(train_loader), args.batch_size)
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        logger.info('Epoch %s/%s started', epoch, args.epochs)
        loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, args.log_every, logger)
        logger.info('Epoch %s: validating %s sites', epoch, len(val_loader.dataset))
        truth, scores = collect_probabilities(model, val_loader, device)
        score = float(f1_score(truth, scores.argmax(axis=1), labels=range(len(LABEL_ORDER)), average='macro'))
        history.append({'epoch': epoch, 'train_loss': loss, 'validation_macro_f1': score,
                        'seconds': time.monotonic() - started})
        logger.info('Epoch %s complete: loss=%.4f, val macro F1=%.4f, %.1fs',
                    epoch, loss, score, history[-1]['seconds'])
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
            save_checkpoint({'model_state_dict': best_state, 'epoch': epoch,
                             'validation_macro_f1': score, 'sequence_length': length,
                             'label_order': LABEL_ORDER}, run_dir / 'best_weights.pt')
        else:
            stale_epochs += 1
        history_path = run_dir / 'training_history.csv.tmp'
        pd.DataFrame(history).to_csv(history_path, index=False)
        history_path.replace(run_dir / 'training_history.csv')
        save_checkpoint({'epoch': epoch, 'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'best_validation_macro_f1': best_score, 'history': history,
                         'sequence_length': length, 'label_order': LABEL_ORDER},
                        run_dir / 'latest_checkpoint.pt')
        if stale_epochs >= args.patience:
            logger.info('Early stopping after %s epochs without improvement', args.patience)
            break
    if best_state is None:
        raise RuntimeError('Training did not produce a valid checkpoint.')
    model.load_state_dict(best_state)
    model.eval()
    return model, pd.DataFrame(history), best_score


def export_bundle(model, history, best_val_f1, args, bundle_dir, frame, splits,
                  holdouts, summary, per_class, check_inputs):
    """
    Exports a new schema-1 CPU bundle and verifies in-memory/disk predictions.

    PARAMETERS
    ----------
    model : torch.nn.Module
        validation-selected model; the original instance stays on its device
    history : pandas.DataFrame
        per-epoch training and validation history
    best_val_f1 : float
        validation score used for checkpoint selection
    args : argparse.Namespace
        identity, resolved feature path, and training settings
    bundle_dir : pathlib.Path
        destination which must not already exist
    frame : pandas.DataFrame
        complete validated input table for provenance/counts
    splits : dict[str, pandas.DataFrame]
        train, validation, and test tables after overlap removal
    holdouts : dict[str, list[str]]
        original validation/test chromosome assignments
    summary : dict
        final held-out test metrics, never used for checkpoint selection
    per_class : pandas.DataFrame
        per-class test metrics
    check_inputs : torch.Tensor
        small CPU validation batch used to verify export parity

    RAISES
    ------
    FileExistsError
        if the bundle directory already exists; research outputs are never reused
    AssertionError
        if exported predictions differ from the selected CPU model
    """
    cpu_model = copy.deepcopy(model).cpu().eval()
    length = check_inputs.shape[2]
    exported = torch.jit.script(cpu_model)
    with torch.inference_mode():
        torch.testing.assert_close(exported(check_inputs), cpu_model(check_inputs))
    bundle_dir.mkdir(parents=True, exist_ok=False)
    artifact = bundle_dir / 'model.ts'
    exported.save(str(artifact))
    restored = torch.jit.load(str(artifact), map_location='cpu').eval()
    with torch.inference_mode():
        torch.testing.assert_close(restored(check_inputs), cpu_model(check_inputs))
    torch.save({'model_state_dict': cpu_model.state_dict(), 'sequence_length': length,
                'label_order': LABEL_ORDER, 'channels': list(BASES)}, bundle_dir / 'weights.pt')
    history.to_csv(bundle_dir / 'training_history.csv', index=False)
    per_class.to_csv(bundle_dir / 'test_per_class.csv', index=False)
    feature_path = args.features
    dataset_path = feature_path.relative_to(Path.cwd()) if feature_path.is_relative_to(Path.cwd()) else feature_path
    metadata = {
        'schema_version': 1,
        'genome_id': args.genome_id, 'species_name': args.species_name,
        'assembly_id': args.assembly_id, 'model_version': args.model_version,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'artifact': 'model.ts', 'artifact_sha256': file_sha256(artifact),
        'format': 'torchscript', 'torch_version': str(torch.__version__),
        'weights': 'weights.pt', 'weights_sha256': file_sha256(bundle_dir / 'weights.pt'),
        'architecture': str(cpu_model),
        'input': {'layout': 'NCL', 'dtype': 'float32', 'channels': list(BASES),
                  'sequence_length': length, 'window': length // 2, 'alphabet': 'ACGT',
                  'orientation': 'transcript', 'ambiguous_bases': 'reject', 'padding': 'none'},
        'output': {'type': 'logits', 'label_order': LABEL_ORDER,
                   'score_transform': 'softmax', 'calibrated': False},
        'coordinates': '1-based anchor in submitted sequence; donor first intronic base, acceptor last intronic base',
        'dataset': {'path': str(dataset_path), 'sha256': file_sha256(feature_path),
                    'class_counts': {str(k): int(v) for k, v in frame['label'].value_counts().items()}},
        'training': {'seed': RANDOM_SEED, 'max_epochs': args.epochs, 'patience': args.patience,
                     'batch_size': args.batch_size, 'learning_rate': args.learning_rate,
                     'torch_cpu_threads': args.cpu_threads, 'best_validation_macro_f1': best_val_f1,
                     'train_chromosomes': sorted(splits['train']['chrom'].unique().tolist()),
                     'validation_chromosomes': holdouts['validation'], 'test_chromosomes': holdouts['test'],
                     'rows': {name: len(partition) for name, partition in splits.items()}},
        'test_metrics': summary,
        'limitations': ['Within-genome evaluation only; not a cross-species model.',
                       'Sampled negatives do not reflect genome-wide splice-site prevalence.',
                       'Sliding-window website predictions require separate scanning validation.'],
    }
    write_json(bundle_dir / 'metadata.json', metadata)


def load_bundle(bundle_dir, device):
    """
    Loads a trusted local schema-1 bundle after checking its contract and hash.

    PARAMETERS
    ----------
    bundle_dir : str or pathlib.Path
        directory containing metadata.json and model.ts; trust must be established
        independently because a matching hash is not a safety guarantee
    device : torch.device
        CPU, CUDA, or MPS device used for inference

    RETURNS
    -------
    tuple[torch.jit.ScriptModule, dict]
        inference model and parsed metadata

    RAISES
    ------
    ValueError
        if schema, preprocessing, output contract, hash, or model shape is invalid
    """
    bundle_dir = Path(bundle_dir)
    info = json.loads((bundle_dir / 'metadata.json').read_text())
    inputs, outputs = info.get('input', {}), info.get('output', {})
    length = inputs.get('sequence_length')
    if (info.get('schema_version') != 1 or info.get('format') != 'torchscript'
            or info.get('artifact') != 'model.ts' or inputs.get('channels') != list(BASES)
            or inputs.get('layout') != 'NCL' or inputs.get('dtype') != 'float32'
            or type(length) is not int or length < 2 or length % 2
            or inputs.get('window') != length // 2 or inputs.get('orientation') != 'transcript'
            or inputs.get('alphabet') != 'ACGT' or inputs.get('ambiguous_bases') != 'reject'
            or inputs.get('padding') != 'none' or outputs.get('label_order') != LABEL_ORDER
            or outputs.get('type') != 'logits' or outputs.get('score_transform') != 'softmax'
            or outputs.get('calibrated') is not False):
        raise ValueError('Unsupported bundle/preprocessing contract.')
    artifact = bundle_dir / 'model.ts'
    if file_sha256(artifact) != info['artifact_sha256']:
        raise ValueError('Model artifact hash mismatch.')
    model = torch.jit.load(str(artifact), map_location=device).eval()
    with torch.inference_mode():
        if model(torch.zeros(1, 4, length, device=device)).shape != (1, 3):
            raise ValueError('Artifact output does not match its manifest.')
    return model, info


# Saved-model CSV evaluation (no splitting or training)

def _dataset_names(paths):
    """Names datasets by parent/stem, adding numeric suffixes for collisions."""
    names, used = [], set()
    for path in paths:
        base = path.parent.name if path.stem == 'features' and path.parent.name else path.stem
        if path.stem != 'features' and path.parent.name:
            base = f'{path.parent.name}__{path.stem}'
        name, n = base, 2
        while name in used:
            name, n = f'{base}_{n}', n + 1
        used.add(name)
        names.append(name)
    return names


def prepare_evaluation_chunk(chunk, length, allow_crop=True):
    """
    Normalizes CSV sequences/labels and marks invalid rows without deleting bases.

    Longer windows may be symmetrically cropped only for even length differences,
    preserving the transcript-oriented anchor at length // 2. Bases outside the
    retained window are not scored. Unknown labels invalidate labeled rows.

    PARAMETERS
    ----------
    chunk : pandas.DataFrame
        sequence column and optional label/provenance columns
    length : int
        model's required even window length
    allow_crop : bool, optional
        whether longer windows may be center-cropped, by default True

    RETURNS
    -------
    tuple
        normalized sequences, label IDs (or None), per-row skip reasons (empty
        when valid), and a boolean mask marking cropped rows
    """
    seqs = chunk['sequence'].fillna('').astype(str).str.strip().str.upper()
    reason = pd.Series('', index=chunk.index, dtype=object)
    lengths = seqs.str.len()
    too_short, too_long = lengths < length, lengths > length
    odd_extra = (lengths - length) % 2 == 1
    cropped = pd.Series(False, index=chunk.index)
    reason[too_short] = f'shorter than model window ({length})'
    if allow_crop:
        reason[~too_short & odd_extra] = 'cannot center-crop (odd length difference)'
        cropped = too_long & ~odd_extra
        for index in seqs.index[cropped]:
            start = (len(seqs.at[index]) - length) // 2
            seqs.at[index] = seqs.at[index][start:start + length]
    else:
        reason[too_long] = f'longer than model window ({length}); cropping disabled'
    reason[reason.eq('') & ~seqs.str.fullmatch(r'[ACGT]+')] = 'non-ACGT bases'
    targets = None
    if 'label' in chunk.columns:
        labels = chunk['label'].fillna('').astype(str).str.strip().str.lower()
        targets = labels.map(LABEL_TO_INT)
        reason[reason.eq('') & targets.isna()] = f'label not in {LABEL_ORDER}'
    return seqs, targets, reason, cropped


def write_evaluation_predictions(rows, sequences, scores, path):
    """Appends original rows and model outputs with stable columns across chunks."""
    predictions = rows.copy()
    # Always include scored_sequence, even when no windows needed cropping.
    predictions['scored_sequence'] = sequences.to_numpy()
    predictions['predicted_class'] = [LABEL_ORDER[i] for i in scores.argmax(axis=1)]
    for i, label in enumerate(LABEL_ORDER):
        predictions[f'score_{label}'] = scores[:, i]
    predictions.to_csv(path, mode='a', header=not path.exists(), index=False)


def report_labeled_evaluation(truth, scores, motif_hits, info, features_csv, out_dir):
    """
    Writes labeled evaluation reports and summarizes class/motif diagnostics.

    PARAMETERS
    ----------
    truth : numpy.ndarray
        class IDs for scored rows only, in prediction order
    scores : numpy.ndarray
        matching (N,3) uncalibrated softmax scores
    motif_hits : dict[str, list[int]]
        canonical motif hit count and total scored sites for donor and acceptor
    info : dict
        bundle metadata containing the model version
    features_csv : pathlib.Path
        source CSV, used to name the confusion plot
    out_dir : pathlib.Path
        this dataset's scratch output directory

    RETURNS
    -------
    tuple[dict, list[str]]
        classification metrics, class counts, motif rates, and per-class summary
        fields; diagnostic notes for absent classes and low canonical motif rates
    """
    metrics, per_class = classification_metrics(truth, scores)
    per_class.to_csv(out_dir / 'per_class.csv', index=False)
    plot_confusion(truth, scores, f"{info['model_version']} on {features_csv.parent.name}",
                   out_dir / 'confusion_matrix.png')
    motif_rates = {label: hits / count if count else None for label, (hits, count) in motif_hits.items()}
    class_counts = {label: int((truth == i).sum()) for i, label in enumerate(LABEL_ORDER)}
    metrics.update({'class_counts': class_counts, 'canonical_motif_rate': motif_rates})
    for row in per_class.to_dict('records'):
        for metric in ('f1', 'average_precision'):
            metrics[f'{row["class"]}_{metric}'] = row[metric]

    notes = []
    for label, rate in motif_rates.items():
        if rate is not None and rate < 0.5:
            notes.append(f'Only {rate:.0%} of {label} sites have the canonical motif at the anchor: '
                         'check strand orientation / anchor convention / window centering.')
    missing = [label for label, count in class_counts.items() if not count]
    if missing:
        notes.append(f'Classes absent from this dataset: {missing}; macro_f1_present excludes them.')
    return metrics, notes


def evaluate_csv(features_csv, out_dir, model, info, device, batch_size, invalid, allow_crop, chunk_rows):
    """
    Streams one CSV into predictions, skipped rows, and optional labeled metrics.

    PARAMETERS
    ----------
    features_csv : pathlib.Path
        CSV with sequences and optional labels; provenance columns are retained
    out_dir : pathlib.Path
        scratch directory owned by this evaluation call
    model : torch.nn.Module
        verified inference model
    info : dict
        schema-1 bundle metadata
    device : torch.device
        model's inference device
    batch_size : int
        number of windows per inference batch
    invalid : str
        skip logs invalid rows; error rejects the entire dataset
    allow_crop : bool
        whether anchor-preserving center cropping is enabled
    chunk_rows : int
        number of CSV rows read at a time; labeled scores still accumulate in memory

    RETURNS
    -------
    dict
        row counts, skip reasons, notes, and optional classification/motif metrics

    RAISES
    ------
    ValueError
        if sequence is missing, no rows are scorable, or invalid rows occur
        under the error policy
    """
    columns = pd.read_csv(features_csv, nrows=0).columns
    if 'sequence' not in columns:
        raise ValueError("Missing required 'sequence' column.")
    length = info['input']['sequence_length']
    pred_path, skip_path = out_dir / 'predictions.csv', out_dir / 'skipped_rows.csv'
    rows = kept = cropped_count = 0
    skip_counts = {}
    y_all, scores_all = [], []
    motif_hits = {'donor': [0, 0], 'acceptor': [0, 0]}
    dtypes = {c: str for c in ('sequence', 'label', 'chrom', 'strand') if c in columns}
    for chunk in pd.read_csv(features_csv, chunksize=chunk_rows, dtype=dtypes):
        rows += len(chunk)
        seqs, targets, reasons, cropped = prepare_evaluation_chunk(chunk, length, allow_crop)
        bad = reasons.ne('')
        if bad.any():
            if invalid == 'error':
                raise ValueError(f'{int(bad.sum())} invalid rows in a chunk (e.g. {reasons[bad].iloc[0]}); '
                                 'use --invalid-sequences skip to drop them.')
            for reason, count in reasons[bad].value_counts().items():
                skip_counts[reason] = skip_counts.get(reason, 0) + int(count)
            chunk.loc[bad].assign(skip_reason=reasons[bad]).to_csv(
                skip_path, mode='a', header=not skip_path.exists(), index=False)
        good = ~bad
        if not good.any():
            continue
        valid_seqs = seqs[good]
        cropped_count += int((cropped & good).sum())
        scores = score_sequences(model, valid_seqs.tolist(), length, batch_size, device)
        kept += len(valid_seqs)
        write_evaluation_predictions(chunk.loc[good], valid_seqs, scores, pred_path)
        if targets is not None:
            y = targets[good].astype(int).to_numpy()
            y_all.append(y)
            scores_all.append(scores)
            for label, motif, offset in (('donor', 'GT', length // 2), ('acceptor', 'AG', length // 2 - 1)):
                selected = valid_seqs[y == LABEL_TO_INT[label]]
                motif_hits[label][0] += int(selected.str.slice(offset, offset + 2).eq(motif).sum())
                motif_hits[label][1] += len(selected)
    if not kept:
        raise ValueError(f'No scorable rows (skip reasons: {skip_counts}).')
    notes = []
    if info.get('dataset', {}).get('sha256') == file_sha256(features_csv):
        notes.append('SAME FILE AS TRAINING DATA: includes training rows; metrics are optimistic.')
    if rows - kept:
        notes.append(f'{rows - kept} rows skipped; see skipped_rows.csv.')
    result = {'rows': rows, 'scored': kept, 'skipped': rows - kept,
              'center_cropped': cropped_count, 'skip_reasons': skip_counts}
    if y_all:
        truth, scores = np.concatenate(y_all), np.concatenate(scores_all)
        metrics, diagnostic_notes = report_labeled_evaluation(
            truth, scores, motif_hits, info, features_csv, out_dir)
        result.update(metrics)
        notes.extend(diagnostic_notes)
    result['notes'] = notes
    model_id = f"{info['genome_id']}/{info['assembly_id']}/{info['model_version']}"
    write_json(out_dir / 'metrics.json', {'model': model_id, 'features': str(features_csv), **result})
    return result


def evaluate_saved_model(bundle_dir, features_csvs, output_dir=None, batch_size=256,
                         device_request='auto', cpu_threads=1, invalid='skip',
                         allow_crop=True, chunk_rows=200_000):
    """
    Scores CSV datasets using one trusted bundle, without training or splitting.

    Each dataset is built in a unique scratch directory and published only after
    success. Existing result directories are never overwritten. Dataset failures
    are recorded in summary.csv and do not stop other datasets.

    PARAMETERS
    ----------
    bundle_dir : str or pathlib.Path
        trusted local schema-1 bundle
    features_csvs : str, pathlib.Path, or iterable[pathlib.Path]
        one or more CSVs with sequence and optional label/provenance columns
    output_dir : str or pathlib.Path, optional
        root for outputs, by default <bundle>/evaluations
    batch_size : int, optional
        inference batch size, by default 256
    device_request : str, optional
        auto (CUDA > MPS > CPU), cpu, cuda, or mps; by default auto
    cpu_threads : int, optional
        Torch CPU thread count, by default the safe single-thread setting
    invalid : str, optional
        skip or error policy for invalid rows, by default skip
    allow_crop : bool, optional
        whether to center-crop longer windows with even length differences
    chunk_rows : int, optional
        CSV read chunk size, by default 200,000

    RETURNS
    -------
    pandas.DataFrame
        one summary row per requested dataset, including status and errors

    RAISES
    ------
    ValueError
        if settings, dataset list, bundle contract, or artifact hash are invalid
    RuntimeError
        if explicit CUDA is unavailable
    """
    if invalid not in ('skip', 'error'):
        raise ValueError('invalid must be skip or error.')
    if min(batch_size, cpu_threads, chunk_rows) < 1:
        raise ValueError('Batch size, CPU threads, and chunk rows must be positive.')
    if isinstance(features_csvs, (str, Path)):
        features_csvs = [features_csvs]
    features_csvs = [Path(path).expanduser().resolve() for path in features_csvs]
    if not features_csvs:
        raise ValueError('Provide at least one features CSV.')
    torch.set_num_threads(cpu_threads)
    device = select_device(device_request)
    bundle_dir = Path(bundle_dir).expanduser().resolve()
    model, info = load_bundle(bundle_dir, device)
    model_id = f"{info['genome_id']}/{info['assembly_id']}/{info['model_version']}"
    root = Path(output_dir).expanduser().resolve() if output_dir is not None else bundle_dir / 'evaluations'
    root.mkdir(parents=True, exist_ok=True)
    print(f"Model: {model_id} | window: {info['input']['sequence_length']} | device: {device}")
    records = []
    for path, name in zip(features_csvs, _dataset_names(features_csvs)):
        print(f'\n=== {name}: {path}')
        record = {'dataset': name, 'features': str(path), 'model': model_id,
                  'evaluated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}
        scratch = None
        try:
            destination = root / name
            if destination.exists():
                raise FileExistsError(f'{destination} exists; use another --eval-output-dir')
            scratch = Path(tempfile.mkdtemp(prefix=f'.{name}-', dir=root))
            result = evaluate_csv(path, scratch, model, info, device, batch_size, invalid, allow_crop, chunk_rows)
            scratch.rename(destination)
            record.update({'status': 'ok', **{key: value for key, value in result.items()
                                             if key not in ('skip_reasons', 'class_counts',
                                                            'canonical_motif_rate', 'notes')}})
            for label, rate in result.get('canonical_motif_rate', {}).items():
                record[f'{label}_motif_rate'] = rate
            record['notes'] = ' | '.join(result['notes'])
            print(f"scored {result['scored']:,}/{result['rows']:,}")
            for note in result['notes']:
                print('  NOTE:', note)
        except Exception as error:
            record.update({'status': 'error', 'notes': f'{type(error).__name__}: {error}'})
            print('  FAILED:', record['notes'])
        finally:
            if scratch is not None and scratch.exists():
                shutil.rmtree(scratch)
        records.append(record)
    current = pd.DataFrame(records)
    summary_path = root / 'summary.csv'
    combined = pd.concat([pd.read_csv(summary_path), current], ignore_index=True) if summary_path.exists() else current
    combined.to_csv(summary_path, index=False)
    print(f'\nSummary ({summary_path}):')
    print(current[[column for column in ('dataset', 'status', 'scored', 'skipped', 'accuracy', 'macro_f1_present')
                   if column in current.columns]].to_string(index=False))
    return current


# Entry points: the training workflow reads in execution order here.

def run_training(args):
    """
    Runs the training data flow with explicit validation and test boundaries.

    Loads and inspects the CSV, holds out chromosomes, removes sequence overlap,
    and prepares batch encoders. With --train, creates a fresh run, selects weights
    on validation only, evaluates test once, then exports the selected model.
    Without --train, performs preparation but creates no output directories.

    PARAMETERS
    ----------
    args : argparse.Namespace
        CLI settings from parse_args; feature/output paths are resolved
        in place relative to the working directory

    RAISES
    ------
    FileNotFoundError
        if the feature CSV is missing
    FileExistsError
        if training would reuse an existing run or bundle directory
    ValueError
        if identity components, dataset contents, or chromosome splits are invalid
    RuntimeError
        if explicit CUDA is unavailable or training produces no valid checkpoint
    """
    args.features = args.features.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    bundle_dir, run_dir = training_paths(args)
    if not args.features.is_file():
        raise FileNotFoundError(f'Features file not found (set --features): {args.features}')
    if args.train:
        if bundle_dir.exists() or run_dir.exists():
            raise FileExistsError('Choose a new model version: bundle or run directory already exists.')
        run_dir.mkdir(parents=True, exist_ok=False)
    logger = setup_logger(run_dir if args.train else None)
    seed_training(args.cpu_threads)
    device = select_device(args.device)
    logger.info('Selected genome: %s %s | version: %s', args.species_name, args.assembly_id, args.model_version)
    logger.info('Reading %s', args.features)
    
    frame, length = load_training_table(args.features)
    inspect_dataset(frame, length, run_dir if args.train else None)
    folds = build_chromosome_folds(frame)
    holdouts = {'test': folds[0], 'validation': folds[1]}
    print('Test chromosomes:', holdouts['test'])
    print('Validation chromosomes:', holdouts['validation'])
    splits = split_by_chromosome(frame, holdouts['test'], holdouts['validation'])
    print(pd.DataFrame({name: table['label'].value_counts().reindex(LABEL_ORDER)
                        for name, table in splits.items()}))
    loaders = make_loaders(splits, length, args.batch_size)
    # Use validation for the preview/export check without advancing training's shuffle.
    inputs, targets = next(iter(loaders['validation']))
    print('Input batch:', tuple(inputs.shape), '| Target batch:', tuple(targets.shape))
    logger.info('PyTorch: %s | CPU threads: %s | device: %s', torch.__version__, torch.get_num_threads(), device)
    if not args.train:
        print('Training/export skipped. Add --train to train and export this genome.')
        return
    logger.info('Live logs and checkpoints: %s', run_dir)
    if device.type == 'cuda':
        logger.info('GPU: %s', torch.cuda.get_device_name(device))
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({'torch_version': str(torch.__version__), 'device': str(device),
                   'seed': RANDOM_SEED, 'sequence_length': length,
                   'test_chromosomes': holdouts['test'], 'validation_chromosomes': holdouts['validation']})
    write_json(run_dir / 'run_config.json', config)
    model, history, best_val_f1 = train_cnn(length, loaders['train'], loaders['validation'],
                                           args, device, run_dir, logger)
    logger.info('Training complete; evaluating %s held-out test sites', len(splits['test']))
    truth, scores = collect_probabilities(model, loaders['test'], device)
    metrics, per_class = classification_metrics(truth, scores)
    model_name = f'{args.genome_id}/{args.model_version}'
    summary = {'model': model_name, 'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1_all']}
    per_class.insert(0, 'model', model_name)
    print(pd.DataFrame([summary]))
    print(per_class)
    plot_confusion(truth, scores, model_name, run_dir / 'test_confusion_matrix.png')
    export_bundle(model, history, best_val_f1, args, bundle_dir, frame, splits,
                  holdouts, summary, per_class, inputs[:8])
    logger.info('Exported: %s', bundle_dir)


def main(argv=None):
    """Dispatches CSV evaluation or the inspect/train/export workflow."""
    args = parse_args(argv)
    if args.evaluate_bundle is not None:
        summary = evaluate_saved_model(
            args.evaluate_bundle, args.eval_features or [args.features], args.eval_output_dir,
            batch_size=max(args.batch_size, 256), device_request=args.device,
            cpu_threads=args.cpu_threads, invalid=args.invalid_sequences, allow_crop=not args.no_crop,
        )
        if summary['status'].ne('ok').any():
            raise SystemExit(1)
    else:
        run_training(args)


if __name__ == '__main__':
    main()
