"""Migrate existing experiment trial metrics from CSV to Parquet.

Run from project root:
    python -m script.experiment.migrate_parquet
    python -m script.experiment.migrate_parquet --root experiment_results

Completed experiments: convert final CSV, then delete CSV and chunk files.
Partial experiments: convert each chunk CSV and preserve resumability.
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import EXPERIMENT_RESULTS_DIR
from script.experiment.record import METRIC_SCHEMA


CSV_DTYPES = {
    "Trial_id": "int64",
    "Position_index": "int64",
    "Metric": "string",
    "Variant_id": "string",
    "Value": "float32",
}


def convert_csv(path: Path):
    """Write and verify a streamed Parquet copy before deleting the source."""
    target = path.with_suffix(".parquet")
    if target.exists():
        raise FileExistsError(f"Both CSV and Parquet exist: {path}")

    temp = target.with_suffix(".tmp")
    rows = 0
    try:
        with pq.ParquetWriter(temp, METRIC_SCHEMA, compression="zstd") as writer:
            for batch in pd.read_csv(path, chunksize=100_000, dtype=CSV_DTYPES):
                writer.write_table(
                    pa.Table.from_pandas(batch, schema=METRIC_SCHEMA, preserve_index=False)
                )
                rows += len(batch)

        if pq.read_metadata(temp).num_rows != rows:
            raise RuntimeError(f"Row count mismatch: {path}")
        temp.replace(target)
    finally:
        if temp.exists():
            temp.unlink()

    path.unlink()
    print(f"  Converted {path.name} -> {target.name} ({rows:,} rows)")


def migrate_experiment(directory: Path):
    """Migrate one experiment, respecting completed versus partial status."""
    final_csv = directory / "trial_metrics.csv"
    final_parquet = directory / "trial_metrics.parquet"
    chunks = directory / "chunks"

    if final_csv.exists():
        convert_csv(final_csv)

    if final_parquet.exists():
        # Only drop chunks once the completed, readable final file exists.
        pq.read_metadata(final_parquet)
        if chunks.exists():
            shutil.rmtree(chunks)
            print("  Removed completed chunks")
    elif chunks.exists():
        for chunk in sorted(chunks.glob("chunk_*.csv")):
            convert_csv(chunk)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=EXPERIMENT_RESULTS_DIR)
    args = parser.parse_args()

    for snapshot in sorted(args.root.rglob("experiment_snapshot.yml")):
        directory = snapshot.parent
        print(directory)
        migrate_experiment(directory)


if __name__ == "__main__":
    main()
