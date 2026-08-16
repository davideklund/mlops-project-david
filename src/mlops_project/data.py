"""Shared column definitions and a small data-loading utility.

This intentionally does NOT include feature engineering, target
construction (e.g. computing remaining useful life), or train/test loading
with a target attached -- that's part of the assignment. As you develop
your approach in exploration.ipynb, add your own preprocessing functions
here (or in a new module under this package) and reuse them from both
train.py and model.py, so training and inference always agree.

`MODEL_NAME` / `MODEL_ALIAS` and `load_scoring_batch` below exist purely so
predict.py / predict_via_rest.py work out of the box regardless of what
preprocessing you build -- they only need to know the model's registered
name and where the raw data files live, not anything about your features.
"""
from pathlib import Path

import pandas as pd

# data/ lives at the project root, as a sibling of src/. This file lives at
# src/mlops_project/data.py, so we go up two levels to reach the project root.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

INDEX_COLUMNS = ["unit_number", "time_cycles"]
SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
COLUMN_NAMES = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS

# Name and alias the model is registered under in the MLflow Model
# Registry -- shared with train.py so predict.py / predict_via_rest.py can
# always find "the current model" by name, without hardcoding version
# numbers. Feel free to rename.
MODEL_NAME = "turbofan_rul_model"
MODEL_ALIAS = "champion"


def _load_raw(path: Path) -> pd.DataFrame:
    """Read a raw CMAPSS data file: 26 whitespace-separated columns, no header."""
    return pd.read_csv(
        path, sep=r"\s+", header=None, names=COLUMN_NAMES, usecols=range(len(COLUMN_NAMES))
    )


def load_scoring_batch(
    dataset: str = "FD001", path: str | None = None, n: int | None = None
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Load a batch of engines' latest readings to score -- used by
    predict.py and predict_via_rest.py to simulate "new data arriving."

    With no `path` given, this defaults to the dataset's test file, in
    which case the matching true RUL values (from RUL_<dataset>.txt) are
    returned too, so you can compare your predictions against them.
    """
    default_test_path = DATA_DIR / f"test_{dataset}.txt"
    load_path = Path(path) if path is not None else default_test_path

    df = _load_raw(load_path)
    last_row_idx = df.groupby("unit_number")["time_cycles"].idxmax()
    df_last = df.loc[last_row_idx].reset_index(drop=True)

    true_rul = None
    if load_path == default_test_path:
        rul_path = DATA_DIR / f"RUL_{dataset}.txt"
        if rul_path.exists():
            true_rul = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["RUL"])["RUL"]

    if n is not None:
        df_last = df_last.head(n)
        if true_rul is not None:
            true_rul = true_rul.head(n)

    return df_last, true_rul
