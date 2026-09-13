"""
Ingestion Runner Script for AmritaGPT.
Parses all institutional documents in data/ and constructs both dense vector and sparse BM25 indices.
"""
import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amritagpt.ingestion.pipeline import IngestionPipeline
from amritagpt.config import DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="AmritaGPT Ingestion Bus")
    parser.add_argument("--force", action="store_true", help="Force reindexing of all documents")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files to process")
    args = parser.parse_args()

    print("=" * 70)
    print("  AMRITAGPT: INSTITUTIONAL INTELLIGENCE INGESTION ENGINE")
    print(f"  Source directory: {DATA_DIR}")
    print(f"  Force reindex: {args.force}")
    print("=" * 70)

    pipeline = IngestionPipeline(data_dir=DATA_DIR)
    results = pipeline.run_ingestion(force_reindex=args.force, max_files=args.max_files)

    print("\n" + "=" * 70)
    print("  INGESTION SUMMARY")
    print("=" * 70)
    for k, v in results.items():
        print(f"  • {k.replace('_', ' ').title()}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    main()
