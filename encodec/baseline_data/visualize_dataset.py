import argparse
import sys
from pathlib import Path

import numpy as np

from encodec.baseline_data import UniversalWrapper
from encodec.trainer.init_config import load_config

# Add the B directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[3] / "time_series_foundation_models/dataloaders"))
from universal_loader import Object  # type: ignore

"""
poetry run python encodec/baseline_data/visualize_dataset.py 
"""


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="baseline_ppg")
    return parser.parse_args()


if __name__ == "__main__":
    args = set_args()

    config_dict, config = load_config(
        filepath=f"encodec/params/{args.config}.yaml",
        schemapath="encodec/params/schema.json",
    )

    # Load dataset
    compression_ratio = np.prod(config.model.ratios)
    dataset_args = Object(
        dataset=config.dataset.datasets,
        mode=config.dataset.mode,
        label="mit_gender",
        seq_len=config.model.sample_rate * config.dataset.max_length,
        fold=config.dataset.cv,
        z_score=True,
        exclude_dataset=None,
        debug=False,
    )
    # create train/val datasets
    train_dataset = UniversalWrapper(
        args=dataset_args,
        type="train",
        compression_ratio=compression_ratio,
        debug_training=False,
    )

    train_dataset.visualize_sample(save_dir=f"dataset_visualizations/{args.config}", num_samples=10)
