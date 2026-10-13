from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pulmoai.preprocessing import preprocess_count_mapping, read_single_profile_csv


FEATURES = ("gene_a", "gene_b", "gene_c")


class CountPreprocessingTests(unittest.TestCase):
    def write_csv(self, headers: list[str], rows: list[list[object]]) -> Path:
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / "profile.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(rows)
        return path

    def test_wide_csv_aligns_shuffled_genes_and_normalizes_counts(self) -> None:
        path = self.write_csv(
            ["sample_id", "gene_c", "gene_a", "gene_b"],
            [["sample-1", 4, 1, 5]],
        )
        observed = read_single_profile_csv(path, FEATURES)
        expected = np.log1p(np.array([[1, 5, 4]], dtype=np.float64) / 10 * 1_000_000).astype(np.float32)
        np.testing.assert_allclose(observed, expected, rtol=1e-6)

    def test_long_csv_is_supported(self) -> None:
        path = self.write_csv(
            ["gene_id", "count"],
            [["gene_c", 4], ["gene_a", 1], ["gene_b", 5]],
        )
        observed = read_single_profile_csv(path, FEATURES)
        self.assertEqual(observed.shape, (1, len(FEATURES)))
        np.testing.assert_allclose(observed[0, 0], np.log1p(100_000), rtol=1e-6)

    def test_missing_gene_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            preprocess_count_mapping({"gene_a": 1, "gene_b": 2}, FEATURES)

    def test_extra_gene_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            preprocess_count_mapping({"gene_a": 1, "gene_b": 2, "gene_c": 3, "extra": 4}, FEATURES)

    def test_duplicate_long_format_gene_is_rejected(self) -> None:
        path = self.write_csv(
            ["gene", "count"],
            [["gene_a", 1], ["gene_a", 2]],
        )
        with self.assertRaisesRegex(ValueError, "duplicate gene"):
            read_single_profile_csv(path, FEATURES)

    def test_multiple_profiles_are_rejected(self) -> None:
        path = self.write_csv(
            list(FEATURES),
            [[1, 2, 3], [4, 5, 6]],
        )
        with self.assertRaisesRegex(ValueError, "multiple profiles"):
            read_single_profile_csv(path, FEATURES)

    def test_negative_fractional_and_nonfinite_counts_are_rejected(self) -> None:
        for bad_values in (
            {"gene_a": -1, "gene_b": 2, "gene_c": 3},
            {"gene_a": 1.5, "gene_b": 2, "gene_c": 3},
            {"gene_a": np.nan, "gene_b": 2, "gene_c": 3},
        ):
            with self.subTest(bad_values=bad_values), self.assertRaises(ValueError):
                preprocess_count_mapping(bad_values, FEATURES)

    def test_zero_count_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonzero"):
            preprocess_count_mapping({"gene_a": 0, "gene_b": 0, "gene_c": 0}, FEATURES)

    def test_duplicate_model_feature_names_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            preprocess_count_mapping({"gene_a": 1, "gene_b": 2}, ("gene_a", "gene_a"))


if __name__ == "__main__":
    unittest.main()
