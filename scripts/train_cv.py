#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physiofuse.dataset import load_label_csv, validate_dataset_layout
from physiofuse.train import run_cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PhysioFuse with stratified cross-validation.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "physiofuse.yaml")
    parser.add_argument("--check-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    df = load_label_csv(cfg["data"]["label_csv"])
    missing = validate_dataset_layout(cfg["data"]["data_dir"], df)
    if missing:
        print("Missing data files, first examples:")
        for item in missing:
            print(f"  {item}")
        raise SystemExit(1)
    if args.check_data:
        print(f"Data layout OK. Samples: {len(df)}")
        return
    summary = run_cv(cfg, df)
    print(f"Saved CV summary to {Path(cfg['output']['out_dir']) / 'cv_results.json'}")
    print(summary)


if __name__ == "__main__":
    main()

