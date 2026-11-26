# CNN Splice

Train a convolutional neural network (CNN) to classify DNA sequence windows as **none**, **donor**, or **acceptor**. CNN Splice builds labeled windows from a reference genome and annotation, trains a genome-specific model with chromosome holdouts, and evaluates saved models on CSV inputs.

**Initial release:** a research baseline, not a biologically validated splice-site predictor. No datasets or pretrained weights are distributed; you must supply data and train a model. Whole-genome scanning and cross-species generalization are not established by this release.

## What is included?

- Dataset construction from matching FASTA and GFF3 files, including background and motif-centered negative examples.
- A small, two-convolution-layer classifier using transcript-oriented DNA sequences.
- Chromosome-separated training, validation-based checkpoint selection, and held-out test metrics.
- Versioned model export and saved-model prediction on labeled or unlabeled CSV windows.
- Synthetic tests for dataset construction, training, export, and evaluation.

The project consists of command-line scripts, not an installable Python package or web service. Genomic scanning, interval evaluation, and multi-organism orchestration are outside the initial release.

## Installation

Use **Python 3.11 or newer** with a compatible PyTorch build. Python 3.12 is a conservative choice for the current TorchScript-based export format.

Run commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in PowerShell instead.

For NVIDIA GPUs, follow the [PyTorch installation instructions](https://pytorch.org/get-started/locally/) to install a compatible CUDA build before installing the requirements. The scripts offer CPU, CUDA, and Apple MPS device selection; CUDA has not been verified for this release. TorchScript is deprecated upstream, and Python 3.14+ may produce compatibility warnings.

## Workflow

The following examples use `my_species` and `assembly_v1` as placeholders. Replace them with your organism and assembly identifiers. There is no bundled example genome, so these commands require your own input files.

### 1. Build labeled windows

Obtain a reference genome in FASTA format and a GFF3 annotation from the **same assembly and release**. Sequence IDs in the annotation must match the FASTA record IDs. The annotation must have transcript/exon relationships from which introns can be inferred.

```bash
python scripts/build_dataset.py \
  --genome /path/to/genome.fna \
  --annotation /path/to/annotation.gff3 \
  --output data/my_species/splice_dataset.csv \
  --window 40 --seed 42 --hard-negative-fraction 0.5 \
  --rebuild-db
```

- `--window 40` extracts **80-base** windows, not 81.
- Transcript features default to `mRNA`; use `--transcript-feature TYPE` for another GFF3 feature type.
- Repeat `--exclude-chrom ID` to exclude unsuitable sequences, such as a mitochondrial genome.
- The builder caches an annotation database next to the output. Use `--rebuild-db` when changing the annotation, or choose a separate cache with `--database`.
- The number of `none` examples equals the combined number of donor and acceptor examples. By default, half the negatives are GT/AG motif decoys and half are background samples. This is a sampled training distribution, not natural genomic prevalence.

Review the printed class counts and canonical-motif diagnostics before training. Keep source URLs, release identifiers, file hashes, exclusions, and the build command with your experiment; the builder does not enforce assembly provenance or independently validate biological labels.

### 2. Train and export

The dataset CSV can be passed directly to the trainer, despite the argument name `--features`:

```bash
python -u scripts/train_cnn.py \
  --features data/my_species/splice_dataset.csv \
  --genome-id my_species --species-name "My species" \
  --assembly-id assembly_v1 --model-version cnn_v001 \
  --device auto --epochs 20 --patience 4 \
  --batch-size 64 --learning-rate 0.001 --train
```

Training requires at least **three chromosomes or contigs**, and all three labels must remain in each split after overlap removal. The trainer automatically groups whole chromosomes into up to five size-balanced folds, using the first for test, the second for validation, and the rest for training. It removes exact-sequence overlap across splits with priority **test > validation > train**; duplicates within a split remain.

Validation macro F1 selects the best checkpoint and controls early stopping. The selected model is then evaluated on the test split. Preserve the CSV and recorded split assignments: changing the data can change the automatic holdouts. Keep test data out of subsequent tuning.

`--device auto` selects CUDA, then Apple MPS, then CPU. Use `--device cpu`, `--device cuda`, or `--device mps` to choose explicitly; an unavailable accelerator causes an error rather than a silent fallback. Exported bundles run on CPU regardless of the training device.

**Choose a new `--model-version` for every run.** Training refuses to reuse an existing run or bundle directory. Identity fields used in paths (`--genome-id`, `--assembly-id`, `--model-version`) accept letters, digits, underscores, and hyphens, beginning with a letter or digit.

Omitting `--train` inspects and prepares the dataset without training or exporting. It still loads, validates, splits, and prepares the full table; it is not a lightweight file check.

#### Optional feature table

```bash
python scripts/build_features.py \
  --input data/my_species/splice_dataset.csv \
  --output data/my_species/features.csv
```

This adds sequence-group identifiers and numeric sequence features. The CNN ignores the numeric feature columns and performs its own one-hot encoding, so this step is **not required** for CNN training. By default, genomic rows are retained; `--deduplicate-sequences` keeps one row per sequence-label pair.

### 3. Predict or evaluate saved models

Only load model bundles from trusted sources.

```bash
python scripts/train_cnn.py \
  --evaluate-bundle artifacts/my_species/assembly_v1/cnn_v001 \
  --eval-features /path/to/independent_windows.csv \
  --eval-output-dir results/cnn_v001_independent \
  --device cpu
```

The CSV needs a `sequence` column. Include `label` to calculate classification metrics; omit it for prediction only. Supply multiple paths after `--eval-features` to score several CSVs. Each input receives a results subdirectory, and the output root contains `summary.csv`.

Outputs include:

- `predictions.csv`: original row data, the scored sequence, predicted class, and `score_none`, `score_donor`, and `score_acceptor`.
- `skipped_rows.csv`: invalid rows and reasons, when present.
- `metrics.json`: processing counts and available metrics.
- `per_class.csv` and `confusion_matrix.png` for labeled inputs.

Scores are softmax outputs, **not calibrated biological probabilities**. Evaluation skips invalid rows by default; use `--invalid-sequences error` to reject them. Longer sequences are center-cropped when the length difference is even; `--no-crop` disables cropping. Short sequences and non-ACGT bases in the scored window are invalid. Check skipped-row counts before interpreting metrics.

Use independent held-out windows for performance estimates. Passing the training CSV to saved-model evaluation does not recreate the held-out test split—it scores the supplied rows, including training examples. This command classifies prepared windows; it does not scan a genome or discover splice sites in arbitrary sequences.

## Input format and coordinates

Training CSVs require these columns:

| Column | Meaning |
| --- | --- |
| `sequence` | Uppercase A/C/G/T DNA, oriented in the direction of transcription. All training windows must share one even length of at least two bases. |
| `label` | Exactly `none`, `donor`, or `acceptor`. Training requires all three. |
| `chrom` | Reference chromosome or contig ID, used for holdouts. |
| `strand` | `+` or `-` on the reference genome. |
| `position` | Original 1-based genomic anchor coordinate. |

Additional columns, including the builder's intron coordinates and `negative_type`, may be retained. Identical training sequences with conflicting labels are rejected.

A donor anchor is the **first intronic base** in transcription order; an acceptor anchor is the **last intronic base**. Minus-strand windows are reverse-complemented. For a window of length `2W`, the anchor is at zero-based sequence index `W`; canonical donor `GT` is at `[W:W+2]`, and acceptor `AG` at `[W-1:W+1]`.

Custom evaluation windows must use the same orientation and centering. Evaluation normalizes sequence case and surrounding whitespace but does not infer strand orientation or move the anchor. Genomic metadata is optional for evaluation and is retained rather than independently verified.

## Training outputs

By default, training creates two directories under `artifacts/` (change the root with `--output-dir`):

```text
artifacts/<genome-id>/<assembly-id>/
  <model-version>_run/   # Logs, configuration, checkpoints, history, diagnostic plots
  <model-version>/       # Exported model bundle
    model.ts
    weights.pt
    metadata.json
    training_history.csv
    test_per_class.csv
```

Keep the bundle files together. `metadata.json` records the input contract, label order, file hashes, dataset hash, chromosome assignments, training settings, and test metrics. Bundles use TorchScript (`model.ts`); `weights.pt` contains the corresponding model state.

Training checkpoints and logs are saved during the run, but the CLI does not provide a resume option. For remote training, `scripts/train_cnn.py` is self-contained: copy it, the input CSV, and install its dependencies on the training host.

## Baseline results and limitations

An initial *Caenorhabditis elegans* WBcel235 run achieved **95.71% macro F1** and **95.66% accuracy** on 96,328 sampled windows from a held-out chromosome. These are results from one experiment, not a reproducibility guarantee or a genome-scanning accuracy estimate. The original annotation/build provenance is incomplete, and the result has not been independently reproduced.

Important limitations:

- Sampled negatives differ from real genomic prevalence; window-level precision/F1 does not establish genome-wide performance.
- Chromosome holdouts and exact-sequence filtering do not eliminate all possible similarity between splits.
- Annotation-derived labels can be incomplete or ambiguous. Software checks alone do not establish biological correctness.
- The baseline is genome-specific; transfer to other assemblies or species requires separate evaluation.
- Longer-context models, architecture comparisons, and real genomic interval evaluation remain future work.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use temporary synthetic data and do not require genome downloads or pretrained weights. They cover dataset construction, encoding, split separation, checkpoint selection, early stopping, bundle validation, CSV evaluation, overwrite protection, and the standalone training workflow. CPU and Apple MPS workflows have synthetic test coverage; MPS tests skip when that backend is unavailable. Development has been done on Apple sillicon and therefore CUDA has not been verified for this release.

## Source layout

```text
scripts/
  build_dataset.py   # FASTA + GFF3 → labeled windows
  build_features.py  # Optional numeric features and sequence groups
  train_cnn.py       # Inspect, train, export, and evaluate CSV windows
tests/               # Synthetic dataset and training-workflow tests
```

Run any script with `--help` for its complete options. Local development copies may contain additional unreleased tools; they are not part of the initial-release workflow described here.

Datasets, annotations, model bundles, experiment results, and local environments are excluded from Git. Back them up separately.

## License

Code is available under the [MIT license](LICENSE). Genome data, annotations, and third-party dependencies retain their own licenses.
