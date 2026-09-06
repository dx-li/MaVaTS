"""Rebuild the small CC-BY-4.0 data extract from the original UCI archive.

Download the URL in data/README.md, then run
python -m examples.air_quality.prepare /path/to/archive.zip.
No network access or ZIP extraction is performed by this script.
"""

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

STATIONS = ("Aotizhongxin", "Changping", "Dingling", "Dongsi")
POLLUTANTS = ("PM2.5", "PM10", "NO2")
ARCHIVE_SHA256 = "b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4"
DATA = Path(__file__).parent / "data"


def prepare(archive_path, destination=DATA):
    """Aggregate observed 2014 readings; missing values stay missing."""
    raw = Path(archive_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Source archive hash mismatch; inspect upstream changes")
    with zipfile.ZipFile(io.BytesIO(raw)) as outer:
        inner = outer.read("PRSA2017_Data_20130301-20170228.zip")
    groups = defaultdict(lambda: [[] for _ in POLLUTANTS])
    with zipfile.ZipFile(io.BytesIO(inner)) as archive:
        for station in STATIONS:
            name = next(n for n in archive.namelist() if f"_{station}_" in n)
            for row in csv.DictReader(io.StringIO(archive.read(name).decode())):
                if row["year"] != "2014":
                    continue
                date = f'2014-{int(row["month"]):02d}-{int(row["day"]):02d}'
                values = groups[date, station, int(row["hour"]) // 12]
                for i, pollutant in enumerate(POLLUTANTS):
                    if row[pollutant] != "NA":
                        value = float(row[pollutant])
                        if value < 0:
                            raise ValueError("Negative concentration")
                        values[i].append(value)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "beijing-2014.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["date", "station", "half_day"]
            + list(POLLUTANTS)
            + [p + "_count" for p in POLLUTANTS]
        )
        for (date, station, half), values in sorted(groups.items()):
            means = [f"{sum(v) / len(v):.6f}" if v else "" for v in values]
            writer.writerow([date, station, half] + means + [len(v) for v in values])
    metadata = {
        "dataset": "Chen, S. (2017). Beijing Multi-Site Air Quality. UCI.",
        "doi": "https://doi.org/10.24432/C5RK5G",
        "license": "CC-BY-4.0",
        "source_url": "https://archive.ics.uci.edu/static/public/501/beijing%2Bmulti%2Bsite%2Bair%2Bquality%2Bdata.zip",
        "source_sha256": ARCHIVE_SHA256,
        "extract_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "stations": STATIONS,
        "pollutants": POLLUTANTS,
        "units": "micrograms per cubic metre",
        "shape": [365, 4, 3, 2],
        "train_end_exclusive": "2014-10-01",
        "transformation": "2014 subset; observed-value half-day means and hourly counts; six decimal places; no imputation in extract",
    }
    (destination / "provenance.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(path, metadata["extract_sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    prepare(parser.parse_args().archive)
