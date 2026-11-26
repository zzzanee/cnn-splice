"""
    File: test_trainer.py
    Description:
        Focused regression checks for encoding, leakage prevention, validation
        selection, trusted-bundle loading, and chunked CSV evaluation.

    Example:
        python -m pytest -q tests/test_trainer.py

    Output:
        Test results; all generated inputs and outputs use temporary fixtures.
"""

import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip('numpy')
pd = pytest.importorskip('pandas')
torch = pytest.importorskip('torch')
pytest.importorskip('sklearn')
pytest.importorskip('matplotlib')

SPEC = importlib.util.spec_from_file_location(
    'standalone_trainer', Path(__file__).resolve().parents[1] / 'scripts' / 'train_cnn.py')
trainer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trainer)
torch.set_num_threads(1)


@pytest.fixture
def table():
    """Provides three chromosomes with all labels and distinct DNA sequences."""
    rows = []
    rng = np.random.default_rng(42)
    for chrom in ('train', 'validation', 'test'):
        for label in trainer.LABEL_ORDER:
            rows.append({'sequence': ''.join(rng.choice(list('ACGT'), 16)),
                         'label': label, 'chrom': chrom, 'strand': '+', 'position': 9})
    return pd.DataFrame(rows)


@pytest.fixture
def bundle(tmp_path):
    """Builds a tiny trusted schema-1 inference bundle without training."""
    directory = tmp_path / 'bundle'
    directory.mkdir()
    torch.jit.script(trainer.make_cnn(16).eval()).save(str(directory / 'model.ts'))
    info = {
        'schema_version': 1, 'format': 'torchscript', 'artifact': 'model.ts',
        'artifact_sha256': trainer.file_sha256(directory / 'model.ts'),
        'genome_id': 'fixture', 'assembly_id': 'test', 'model_version': 'tiny',
        'input': {'layout': 'NCL', 'dtype': 'float32', 'channels': list(trainer.BASES),
                  'sequence_length': 16, 'window': 8, 'orientation': 'transcript',
                  'alphabet': 'ACGT', 'ambiguous_bases': 'reject', 'padding': 'none'},
        'output': {'type': 'logits', 'label_order': trainer.LABEL_ORDER,
                   'score_transform': 'softmax', 'calibrated': False},
    }
    trainer.write_json(directory / 'metadata.json', info)
    return directory, info


@pytest.mark.parametrize('cuda,mps,expected', [
    (True, True, 'cuda'), (True, False, 'cuda'),
    (False, True, 'mps'), (False, False, 'cpu'),
])
def test_auto_device_priority(monkeypatch, cuda, mps, expected):
    """Auto prefers CUDA, then MPS, then CPU without allocating device tensors."""
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: mps)
    assert trainer.select_device('auto').type == expected
    assert trainer.select_device('cpu').type == 'cpu'


@pytest.mark.parametrize('device', ['cuda', 'mps'])
@pytest.mark.parametrize('available', [True, False])
def test_explicit_accelerator_selection(monkeypatch, device, available):
    """Explicit accelerators never silently fall back when unavailable."""
    backend = torch.cuda if device == 'cuda' else torch.backends.mps
    monkeypatch.setattr(backend, 'is_available', lambda: available)
    if available:
        assert trainer.select_device(device).type == device
    else:
        with pytest.raises(RuntimeError, match=f'{device.upper()} requested but unavailable'):
            trainer.select_device(device)


def test_device_argument_validation():
    """Accepts MPS through the trainer CLI and rejects unknown device requests."""
    assert trainer.parse_args(['--device', 'mps']).device == 'mps'
    with pytest.raises(ValueError, match='Device must be'):
        trainer.select_device('unknown')


def test_encoding_layout():
    """Checks channel order, positions, dtype, and contiguous batch layout."""
    encoded = trainer.encode_sequences(['ACGT', 'TGCA'], 4)
    assert encoded.shape == (2, 4, 4)
    assert encoded.dtype == np.float32
    assert encoded.flags.c_contiguous
    np.testing.assert_array_equal(encoded[0], np.eye(4, dtype=np.float32))
    np.testing.assert_array_equal(encoded[1], np.eye(4, dtype=np.float32)[::-1])


@pytest.mark.parametrize('sequences', [[], ['ACG'], ['ACGN'], ['acgt'], ['ACGé']])
def test_encoding_rejects_invalid_sequences(sequences):
    """Rejects empty collections, wrong lengths, and noncanonical bases."""
    with pytest.raises(ValueError):
        trainer.encode_sequences(sequences, 4)


def test_chromosome_splits_remove_overlaps_in_priority_order(table):
    """Retains test over validation over train, without deduplicating within a split."""
    shared_test = table.loc[table.chrom.eq('test')].iloc[0].to_dict()
    shared_val = table.loc[table.chrom.eq('validation')].iloc[1].to_dict()
    extra = [dict(shared_test, chrom='train'), dict(shared_test, chrom='validation'),
             dict(shared_val, chrom='train'), dict(shared_test)]
    table = pd.concat([table, pd.DataFrame(extra)], ignore_index=True)
    splits = trainer.split_by_chromosome(table, ['test'], ['validation'])
    assert {key: len(value) for key, value in splits.items()} == {'train': 3, 'validation': 3, 'test': 4}
    for left, right in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        assert not set(splits[left].sequence) & set(splits[right].sequence)
        assert not set(splits[left].chrom) & set(splits[right].chrom)
    folds = trainer.build_chromosome_folds(table)
    assert folds == trainer.build_chromosome_folds(table)
    assert sorted(chrom for fold in folds for chrom in fold) == ['test', 'train', 'validation']


@pytest.mark.parametrize('test_chroms,val_chroms', [
    (['test'], ['test']), (['absent'], ['validation']), ([], ['validation']),
])
def test_invalid_holdouts_fail(table, test_chroms, val_chroms):
    """Fails on overlapping, unknown, and empty holdout assignments."""
    with pytest.raises(ValueError):
        trainer.split_by_chromosome(table, test_chroms, val_chroms)


def test_missing_class_after_overlap_removal_fails(table):
    """Does not silently continue when leakage removal empties a class."""
    table.loc[table.chrom.eq('train') & table.label.eq('none'), 'sequence'] = (
        table.loc[table.chrom.eq('test') & table.label.eq('none'), 'sequence'].iloc[0])
    with pytest.raises(ValueError, match='missing labels'):
        trainer.split_by_chromosome(table, ['test'], ['validation'])


def test_validation_rejects_conflicting_labels(table):
    """Rejects contradictory targets for identical input sequences."""
    conflict = table.iloc[0].copy()
    conflict['label'] = 'donor'
    with pytest.raises(ValueError, match='conflicting'):
        trainer.validate_dataset(pd.concat([table, conflict.to_frame().T], ignore_index=True))


def test_missing_classes_have_explicit_metrics():
    """Distinguishes all-class macro F1 and avoids undefined average precision."""
    metrics, per_class = trainer.classification_metrics([0, 0], [[1, 0, 0], [1, 0, 0]])
    assert metrics['accuracy'] == metrics['macro_f1_present'] == 1
    assert metrics['macro_f1_all'] == pytest.approx(1 / 3)
    assert per_class.average_precision.isna().all()
    with pytest.raises(ValueError, match='softmax'):
        trainer.classification_metrics([0], [[3, 0, 0]])


def test_validation_selects_best_weights_and_stops_early(table, tmp_path, monkeypatch):
    """Restores epoch one's weights when epoch two has worse validation macro F1."""
    splits = trainer.split_by_chromosome(table, ['test'], ['validation'])
    loaders = trainer.make_loaders(splits, 16, 3)
    args = SimpleNamespace(epochs=5, patience=1, learning_rate=0.001, batch_size=3, log_every=100)
    monkeypatch.setattr(trainer, 'make_cnn', lambda length: torch.nn.Linear(1, 1))
    epochs = []

    def fake_train_epoch(model, *unused):
        """Marks each epoch's weights so checkpoint restoration can be observed."""
        epochs.append(len(epochs) + 1)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(epochs[-1])
        return 1.0

    validation_scores = iter([np.eye(3), np.roll(np.eye(3), 1, axis=1)])
    monkeypatch.setattr(trainer, 'train_epoch', fake_train_epoch)
    monkeypatch.setattr(trainer, 'collect_probabilities',
                        lambda *unused: (np.arange(3), next(validation_scores)))
    model, history, best = trainer.train_cnn(
        16, loaders['train'], loaders['validation'], args, torch.device('cpu'), tmp_path,
        logging.getLogger('test-trainer'))
    assert epochs == [1, 2]
    assert len(history) == 2
    assert best == 1
    assert not model.training
    for parameter in model.parameters():
        torch.testing.assert_close(parameter, torch.ones_like(parameter))
    checkpoint = torch.load(tmp_path / 'best_weights.pt', weights_only=True)
    assert checkpoint['epoch'] == 1
    assert (tmp_path / 'latest_checkpoint.pt').is_file()


def test_chunk_preparation_preserves_anchor():
    """Normalizes case and symmetrically crops only even length differences."""
    center = 'ACGTACGTACGTACGT'
    frame = pd.DataFrame({'sequence': [f' aa{center.lower()}tt ', 'A' + center, center[:-1], 'N' * 16],
                          'label': [' Donor ', 'none', 'none', 'acceptor']})
    seqs, targets, reasons, cropped = trainer.prepare_evaluation_chunk(frame, 16)
    assert seqs.iloc[0] == center
    assert targets.iloc[0] == 1
    assert reasons.eq('').tolist() == [True, False, False, False]
    assert cropped.tolist() == [True, False, False, False]
    _, _, reasons, _ = trainer.prepare_evaluation_chunk(frame, 16, allow_crop=False)
    assert 'cropping disabled' in reasons.iloc[0]


def test_chunked_evaluation_has_stable_columns_and_counts(bundle, tmp_path):
    """Mixes cropped/uncropped chunks and excludes invalid crops from scored totals."""
    directory, _ = bundle
    csv = tmp_path / 'mixed.csv'
    center = 'ACGTACGTACGTACGT'
    pd.DataFrame({'sequence': [center, 'AA' + center + 'TT', 'AA' + center + 'TT', center],
                  'label': ['none', 'donor', 'unknown', 'acceptor'],
                  'position': [9, 11, 11, 9]}).to_csv(csv, index=False)
    result = trainer.evaluate_saved_model(directory, [csv], tmp_path / 'results',
                                           device_request='cpu', chunk_rows=1)
    assert result.status.tolist() == ['ok']
    assert result.scored.iloc[0] == 3
    assert result.skipped.iloc[0] == 1
    assert result.center_cropped.iloc[0] == 1
    destination = next((tmp_path / 'results').glob('*/predictions.csv'))
    predictions = pd.read_csv(destination)
    assert len(predictions) == 3
    assert predictions.scored_sequence.tolist() == [center] * 3
    assert predictions.position.tolist() == [9, 11, 9]
    assert not predictions.isna().any().any()
    assert (destination.parent / 'skipped_rows.csv').is_file()


@pytest.mark.parametrize('chunk_rows', [1, 10])
def test_labeled_reports_use_only_scored_rows(bundle, tmp_path, chunk_rows):
    """Preserves metrics, motif warnings, and training-file notes across chunk sizes."""
    directory, info = bundle
    csv = tmp_path / 'labeled.csv'
    pd.DataFrame({'sequence': ['A' * 16, 'C' * 16, 'N' * 16],
                  'label': ['donor', 'donor', 'acceptor']}).to_csv(csv, index=False)
    info['dataset'] = {'sha256': trainer.file_sha256(csv)}
    trainer.write_json(directory / 'metadata.json', info)
    root = tmp_path / 'results'
    result = trainer.evaluate_saved_model(directory, [csv], root, device_request='cpu',
                                          chunk_rows=chunk_rows)
    assert result.status.tolist() == ['ok']
    predictions_path = next(root.glob('*/predictions.csv'))
    predictions = pd.read_csv(predictions_path)
    scores = predictions[[f'score_{label}' for label in trainer.LABEL_ORDER]].to_numpy()
    expected, _ = trainer.classification_metrics([1, 1], scores)
    report_dir = predictions_path.parent
    report = json.loads((report_dir / 'metrics.json').read_text())
    for metric, value in expected.items():
        assert report[metric] == pytest.approx(value)
        assert result[metric].iloc[0] == pytest.approx(value)
    assert report['class_counts'] == {'none': 0, 'donor': 2, 'acceptor': 0}
    assert report['canonical_motif_rate'] == {'donor': 0, 'acceptor': None}
    assert report['scored'] == 2 and report['skipped'] == 1
    assert any('SAME FILE AS TRAINING DATA' in note for note in report['notes'])
    assert any('Only 0% of donor' in note for note in report['notes'])
    assert any("Classes absent from this dataset: ['none', 'acceptor']" in note for note in report['notes'])
    per_class = pd.read_csv(report_dir / 'per_class.csv')
    assert per_class.support.tolist() == [0, 2, 0]
    assert per_class.average_precision.isna().all()
    assert (report_dir / 'confusion_matrix.png').is_file()


def test_unlabeled_evaluation_continues_after_failed_dataset(bundle, tmp_path):
    """Publishes successful datasets only and cleans up its own failed scratch folder."""
    directory, _ = bundle
    invalid = tmp_path / 'invalid.csv'
    valid = tmp_path / 'valid.csv'
    pd.DataFrame({'sequence': ['N' * 16]}).to_csv(invalid, index=False)
    pd.DataFrame({'sequence': ['acgt' * 4], 'chrom': ['original']}).to_csv(valid, index=False)
    root = tmp_path / 'results'
    result = trainer.evaluate_saved_model(directory, [invalid, valid], root, device_request='cpu', invalid='error')
    assert result.status.tolist() == ['error', 'ok']
    assert len(list(root.iterdir())) == 2  # Only successful dataset and summary.csv.
    predictions = next(root.glob('*/predictions.csv'))
    assert pd.read_csv(predictions).chrom.tolist() == ['original']
    assert not (predictions.parent / 'per_class.csv').exists()


@pytest.mark.parametrize('corruption', ['hash', 'channels', 'schema', 'label_order'])
def test_bundle_rejects_hash_or_contract_mismatch(bundle, corruption):
    """Refuses altered artifacts and incompatible preprocessing/output contracts."""
    directory, info = bundle
    if corruption == 'hash':
        with (directory / 'model.ts').open('ab') as handle:
            handle.write(b'changed')
    elif corruption == 'channels':
        info['input']['channels'] = ['T', 'G', 'C', 'A']
    elif corruption == 'schema':
        info['schema_version'] = 999
    else:
        info['output']['label_order'] = ['donor', 'none', 'acceptor']
    (directory / 'metadata.json').write_text(json.dumps(info))
    with pytest.raises(ValueError, match='hash|contract'):
        trainer.load_bundle(directory, torch.device('cpu'))


@pytest.mark.parametrize('argv', [['--cpu-threads', '0'], ['--batch-size', '0'],
                                  ['--learning-rate', 'nan'], ['--train', '--evaluate-bundle', 'bundle']])
def test_cli_rejects_invalid_settings(argv):
    """Rejects invalid numerical settings and mutually exclusive workflow modes."""
    with pytest.raises(SystemExit):
        trainer.parse_args(argv)
