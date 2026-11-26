"""
    File: test_dataset.py
    Description:
        Checks the FASTA/GFF3-to-dataset-to-feature workflow with synthetic
        temporary fixtures. No external datasets are required. Tests are skipped
        when dataset dependencies are unavailable.

    Example:
        python -m pytest -q tests/test_dataset.py

    Output:
        Test results; generated genomes, annotations, databases, and CSVs are
        confined to a temporary directory that is removed after the test.
"""

import csv
import importlib.util
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

HAVE_DATA = all(importlib.util.find_spec(name) for name in ("Bio", "gffutils", "pandas", "numpy"))
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@unittest.skipUnless(HAVE_DATA, "requires dataset dependencies")
class DatasetTests(unittest.TestCase):
    """Checks dataset and feature generation with synthetic inputs."""

    def test_build_both_strands_and_prepare_features(self):
        """Checks strand/coordinate conventions, reproducibility, and provenance retention."""
        from Bio.Seq import Seq

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = random.Random(42)
            genomes = {f"chr{i}": "".join(rng.choices("ACGT", k=2000)) for i in range(3)}
            (root / "genome.fa").write_text("".join(f">{chrom}\n{seq}\n" for chrom, seq in genomes.items()))
            lines = ["##gff-version 3"]
            for chrom in genomes:
                for strand, start, end in (("+", 101, 400), ("-", 601, 900)):
                    ident = f"{chrom}_{'plus' if strand == '+' else 'minus'}"
                    lines.append(f"{chrom}\ttest\tmRNA\t{start}\t{end}\t.\t{strand}\t.\tID={ident}")
                    for n, (a, b) in enumerate(((start, start + 99), (end - 99, end))):
                        lines.append(f"{chrom}\ttest\texon\t{a}\t{b}\t.\t{strand}\t.\tID={ident}_e{n};Parent={ident}")
            (root / "annotation.gff3").write_text("\n".join(lines) + "\n")

            def run(script, *args):
                """Runs a dataset workflow script against temporary fixtures."""
                result = subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, args)],
                                        cwd=root, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            dataset = root / "dataset.csv"
            run("build_dataset.py", "--genome", root / "genome.fa", "--annotation", root / "annotation.gff3",
                "--output", dataset, "--window", "8", "--seed", "42", "--rebuild-db")
            with dataset.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["label"] for row in rows}, {"donor", "acceptor", "none"})
            positives = [row for row in rows if row["label"] != "none"]
            self.assertEqual(len(positives), 12)
            self.assertEqual(sum(row["label"] == "none" for row in rows), 12)
            self.assertEqual({row["strand"] for row in positives}, {"+", "-"})
            for row in rows:
                self.assertEqual(len(row["sequence"]), 16)
                sequence = genomes[row["chrom"]]
                position = int(row["position"])
                if row["strand"] == "+":
                    expected = sequence[position - 1 - 8:position - 1 + 8]
                else:
                    expected = str(Seq(sequence[position - 8:position + 8]).reverse_complement())
                self.assertEqual(row["sequence"], expected)

            original = dataset.read_bytes()
            run("build_dataset.py", "--genome", root / "genome.fa", "--annotation", root / "annotation.gff3",
                "--output", dataset, "--window", "8", "--seed", "42", "--rebuild-db")
            self.assertEqual(dataset.read_bytes(), original)
            features = root / "features.csv"
            run("build_features.py", "--input", dataset, "--output", features)
            with features.open() as handle:
                prepared = list(csv.DictReader(handle))
            self.assertEqual(len(prepared), len(rows))
            for source, target in zip(rows, prepared):
                for column in ("sequence", "label", "chrom", "strand", "position"):
                    self.assertEqual(source[column], target[column])
                self.assertTrue(target["sequence_group"])
