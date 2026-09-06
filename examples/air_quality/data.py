"""Offline loading and training-only preprocessing of observed concentrations."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"


def load():
    """Return dates, (day, station, pollutant, half-day) values and counts."""
    metadata = json.loads((DATA / "provenance.json").read_text())
    path = DATA / "beijing-2014.csv"
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["extract_sha256"]:
        raise ValueError("Bundled data checksum mismatch")
    dates = np.arange("2014-01-01", "2015-01-01", dtype="datetime64[D]")
    values = np.full((365, 4, 3, 2), np.nan)
    counts = np.zeros(values.shape, dtype=int)
    seen = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            t = int((np.datetime64(row["date"]) - dates[0]).astype(int))
            station = metadata["stations"].index(row["station"])
            half = int(row["half_day"])
            key = (t, station, half)
            if key in seen:
                raise ValueError("Duplicate half-day")
            seen.add(key)
            for j, pollutant in enumerate(metadata["pollutants"]):
                values[t, station, j, half] = float(row[pollutant] or "nan")
                counts[t, station, j, half] = int(row[pollutant + "_count"])
    if len(seen) != 365 * 4 * 2:
        raise ValueError("Incomplete calendar")
    return dates, values, counts


def preprocess(values, counts, train=273, tensor=False):
    """Mask sparse observations; impute from training medians, then log/scale.

    Daily means weight half-day means by available hourly counts. A daily cell
    needs 18 readings; a half-day cell needs 9. Imputed targets are NEVER scored.
    The same training-derived scalar scales all cells after cellwise centering.
    This is a teaching transformation, not a stationarity or separability test.
    """
    if tensor:
        raw = values.copy()
        observed = counts >= 9
    else:
        total = counts.sum(axis=-1)
        raw = np.divide(
            (np.nan_to_num(values) * counts).sum(axis=-1),
            total,
            out=np.full(total.shape, np.nan),
            where=total > 0,
        )
        observed = total >= 18
    raw[~observed] = np.nan
    medians = np.nanmedian(raw[:train], axis=0)
    if not np.isfinite(medians).all():
        raise ValueError("A cell has no usable training observations")
    filled = np.where(observed, raw, medians)
    logged = np.log1p(filled)
    mean = logged[:train].mean(axis=0)
    scale = float(np.std(logged[:train] - mean))
    if scale <= 0:
        raise ValueError("Constant training data")
    return (
        (logged - mean) / scale,
        observed,
        {
            "train": train,
            "median": medians,
            "mean": mean,
            "scale": scale,
            "imputed_training_cells": int((~observed[:train]).sum()),
            "imputed_test_cells": int((~observed[train:]).sum()),
        },
    )
