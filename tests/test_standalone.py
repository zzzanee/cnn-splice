"""
    File: test_standalone.py
    Description:
        Exercises training, export, reload, and CSV evaluation using only a
        copied trainer and synthetic temporary data, with an empty PYTHONPATH.
        This checks standalone wiring, not biological accuracy. Tests are skipped
        when training dependencies are unavailable.

    Example:
        python -m pytest -q tests/test_standalone.py

    Output:
        Test results; synthetic CSVs, model bundles, and evaluation reports are
        confined to a temporary directory that is removed after the test.
"""

import csv
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest

HAVE_TRAIN = all(importlib.util.find_spec(name) for name in ("torch", "pandas", "sklearn", "matplotlib"))


@unittest.skipUnless(HAVE_TRAIN, "requires train extras")
class StandaloneTests(unittest.TestCase):
    """Checks the trainer in isolation from the rest of the repository."""

    def test_single_file_train_export_reload_and_evaluate(self):
        """Checks isolated CPU training, bundle round trips and overwrite guards."""
        self.check_train_export_reload_and_evaluate('cpu')

    def test_mps_train_export_reload_and_evaluate(self):
        """Exercises real MPS training and portable CPU export on compatible hosts."""
        import torch
        if not torch.backends.mps.is_available():
            self.skipTest('requires an available MPS device')
        self.check_train_export_reload_and_evaluate('mps')

    def check_train_export_reload_and_evaluate(self, device):
        """Runs the same standalone workflow on the requested device."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "train_cnn.py"
            shutil.copy2(Path(__file__).resolve().parents[1] / "scripts" / "train_cnn.py", script)
            features = root / "features.csv"
            rng = random.Random(18)
            with features.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("sequence", "label", "chrom", "strand", "position"))
                for chrom in ("chr1", "chr2", "chr3"):
                    for label in ("none", "donor", "acceptor"):
                        for _ in range(2):
                            sequence = "".join(rng.choices("ACGT", k=16))
                            writer.writerow((sequence, label, chrom, "+", 9))
            env = dict(os.environ, PYTHONPATH="", MPLCONFIGDIR=str(root / "mpl"))

            def run(*args, expected=0):
                """Runs the copied trainer with an empty PYTHONPATH in a temporary folder."""
                result = subprocess.run([sys.executable, str(script), *map(str, args)], cwd=root,
                                        env=env, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                return result

            run("--features", features, "--output-dir", root / "inspection", "--device", "cpu")
            self.assertFalse((root / "inspection").exists())
            run("--train", "--features", features, "--output-dir", root / "artifacts",
                "--genome-id", "fixture", "--assembly-id", "test", "--model-version", "tiny",
                "--species-name", "Synthetic", "--epochs", "1", "--device", device)
            bundle = root / "artifacts" / "fixture" / "test" / "tiny"
            import torch
            info = json.loads((bundle / "metadata.json").read_text())
            self.assertEqual(info["schema_version"], 1)
            artifact = bundle / info["artifact"]
            self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(),
                             info["artifact_sha256"])
            torch.set_num_threads(1)
            model = torch.jit.load(str(artifact), map_location="cpu").eval()
            with torch.inference_mode():
                scores = model(torch.zeros((1, 4, 16))).softmax(dim=1)
            self.assertEqual(tuple(scores.shape), (1, 3))
            self.assertAlmostEqual(float(scores.sum()), 1.0, places=6)
            run("--evaluate-bundle", bundle, "--eval-features", features,
                "--eval-output-dir", root / "evaluation", "--device", device)
            with (root / "evaluation" / "summary.csv").open() as handle:
                self.assertEqual(next(csv.DictReader(handle))["status"], "ok")
            self.assertEqual(hashlib.sha256((bundle / "weights.pt").read_bytes()).hexdigest(),
                             info["weights_sha256"])
            self.assertEqual(info["input"]["channels"], ["A", "C", "G", "T"])
            self.assertEqual(info["output"]["label_order"], ["none", "donor", "acceptor"])
            self.assertFalse(info["output"]["calibrated"])
            self.assertEqual(set(info["test_metrics"]), {"model", "accuracy", "macro_f1"})
            self.assertTrue((bundle.parent / "tiny_run" / "test_confusion_matrix.png").is_file())
            predictions = next((root / "evaluation").glob("*/predictions.csv"))
            original_predictions = predictions.read_bytes()
            original_metadata = (bundle / "metadata.json").read_bytes()
            run("--evaluate-bundle", bundle, "--eval-features", features,
                "--eval-output-dir", root / "evaluation", "--device", device, expected=1)
            self.assertEqual(predictions.read_bytes(), original_predictions)
            with (root / "evaluation" / "summary.csv").open() as handle:
                self.assertEqual([row["status"] for row in csv.DictReader(handle)], ["ok", "error"])
            run("--train", "--features", features, "--output-dir", root / "artifacts",
                "--genome-id", "fixture", "--assembly-id", "test", "--model-version", "tiny",
                "--epochs", "1", "--device", device, expected=1)
            self.assertEqual((bundle / "metadata.json").read_bytes(), original_metadata)
