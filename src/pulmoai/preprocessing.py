"""Strict input validation and training-matched expression preprocessing."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


INTEGER_COUNT_TOLERANCE = 1e-5


def _validate_feature_order(feature_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(str(name) for name in feature_names)
    if not names or any(not name.strip() for name in names):
        raise ValueError("the model feature list is empty or contains blank identifiers")
    if len(set(names)) != len(names):
        raise ValueError("the model feature list contains duplicate identifiers")
    return names


def _normalize_raw_counts(values: Sequence[object], feature_count: int) -> np.ndarray:
    try:
        counts = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("expression values must be numeric raw counts") from error
    if counts.ndim != 1 or counts.size != feature_count:
        raise ValueError(f"expected one profile with exactly {feature_count} gene counts")
    if not np.isfinite(counts).all():
        raise ValueError("raw counts contain missing or non-finite values")
    if np.any(counts < 0):
        raise ValueError("raw counts cannot be negative")
    rounded_counts = np.rint(counts)
    if np.any(np.abs(counts - rounded_counts) > INTEGER_COUNT_TOLERANCE):
        raise ValueError("values are not integer-like raw counts; normalized expression is unsupported")
    total_counts = float(rounded_counts.sum(dtype=np.float64))
    if not np.isfinite(total_counts) or total_counts <= 0:
        raise ValueError("the profile must contain at least one nonzero raw count")
    return np.log1p(rounded_counts / total_counts * 1_000_000).astype(np.float32)[None, :]


def preprocess_count_mapping(
    values_by_gene: Mapping[str, object],
    feature_names: Sequence[str],
) -> np.ndarray:
    """Align a complete gene-to-count mapping, then apply CPM and log1p."""
    expected = _validate_feature_order(feature_names)
    if any(not isinstance(name, str) for name in values_by_gene):
        raise ValueError("gene identifiers must be strings matching the model feature names exactly")
    supplied = set(values_by_gene)
    missing = [name for name in expected if name not in supplied]
    extra = sorted(supplied.difference(expected))
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {len(missing)} required genes (examples: {missing[:5]})")
        if extra:
            details.append(f"found {len(extra)} unsupported genes (examples: {extra[:5]})")
        raise ValueError("gene identifiers do not match the model: " + "; ".join(details))
    return _normalize_raw_counts([values_by_gene[name] for name in expected], len(expected))


def read_single_profile_csv(path: str | Path, feature_names: Sequence[str]) -> np.ndarray:
    """Read one complete raw-count profile from strict wide or long CSV."""
    expected = _validate_feature_order(feature_names)
    source = Path(path)
    try:
        handle = source.open("r", encoding="utf-8-sig", newline="")
    except OSError as error:
        raise ValueError(f"could not open input CSV: {error}") from error

    with handle:
        reader = csv.reader(handle)
        try:
            headers = next(reader)
        except StopIteration as error:
            raise ValueError("input CSV is empty") from error
        if not headers or any(not header.strip() for header in headers):
            raise ValueError("CSV headers must be nonblank gene identifiers or supported field names")
        if len(set(headers)) != len(headers):
            raise ValueError("CSV contains duplicate column names")

        if len(headers) == 2 and "count" in headers and ("gene" in headers or "gene_id" in headers):
            identifier_column = "gene" if "gene" in headers else "gene_id"
            count_column = "count"
            values_by_gene: dict[str, str] = {}
            for row_number, row in enumerate(reader, start=2):
                if len(row) != 2:
                    raise ValueError(f"malformed long-format CSV row {row_number}")
                gene = row[headers.index(identifier_column)].strip()
                value = row[headers.index(count_column)].strip()
                if not gene:
                    raise ValueError(f"blank gene identifier on CSV row {row_number}")
                if gene in values_by_gene:
                    raise ValueError(f"duplicate gene identifier in CSV: {gene}")
                values_by_gene[gene] = value
            return preprocess_count_mapping(values_by_gene, expected)

        wide_headers = headers.copy()
        sample_id_index = None
        if "sample_id" in wide_headers and "sample_id" not in expected:
            sample_id_index = wide_headers.index("sample_id")
            wide_headers.pop(sample_id_index)
        if set(wide_headers) != set(expected) or len(wide_headers) != len(expected):
            missing = [name for name in expected if name not in set(wide_headers)]
            extra = sorted(set(wide_headers).difference(expected))
            raise ValueError(
                "wide CSV must contain exactly the model's gene identifiers"
                + " (an optional sample_id column is allowed)"
                + f"; missing {len(missing)} genes, unsupported columns {extra[:5]}"
            )
        try:
            row = next(reader)
        except StopIteration as error:
            raise ValueError("wide CSV must contain exactly one profile row") from error
        if len(row) != len(headers):
            raise ValueError("wide CSV profile row does not match its header")
        try:
            next(reader)
        except StopIteration:
            pass
        else:
            raise ValueError("multiple profiles are not supported; provide exactly one profile")

        values_by_gene = {
            header: row[index]
            for index, header in enumerate(headers)
            if index != sample_id_index
        }
        return preprocess_count_mapping(values_by_gene, expected)
