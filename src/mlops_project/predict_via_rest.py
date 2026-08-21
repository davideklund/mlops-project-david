"""Score engines by calling a *served* model over HTTP, instead of loading it
in-process (see predict.py for the in-process version).

This simulates how a real client would consume the model in production: as
a REST endpoint, callable from any language or system, not just Python code
that happens to have `mlflow` installed. That's the actual "deployment"
step -- the model becomes a service, rather than a file you load.

Prerequisite -- start the model server in a separate terminal first:

    export MLFLOW_TRACKING_URI=http://127.0.0.1:8880   # match whatever you set for train.py
    uv run mlflow models serve \\
        --model-uri "models:/turbofan_rul_model@champion" \\
        --port 5001 \\
        --env-manager=local

Then, in this terminal:

    uv run predict-rest
    uv run predict-rest --n 10
    uv run predict-rest --port 5001
    uv run predict-rest --dataset FD003 --n 10   # score + compare against FD003's ground truth
"""
import argparse
from typing import Any

import pandas as pd
import requests

from .data import load_scoring_batch


def _print_results(input_df: pd.DataFrame, predictions: "pd.Series | list | Any", true_rul: pd.Series | None) -> None:
    """Print a table of predicted (and, if available, true) RUL per engine."""
    result = pd.DataFrame({
        "unit_number": input_df["unit_number"],
        "predicted_RUL": pd.Series(predictions).round(1).values,
    })
    if true_rul is not None:
        result["true_RUL"] = true_rul.values
        result["abs_error"] = (result["predicted_RUL"] - result["true_RUL"]).abs().round(1)
    print(result.to_string(index=False))


def main() -> None:
    """CLI entry point (registered as `predict-rest` in pyproject.toml):
    parse arguments, POST a batch of engines to an already-running
    `mlflow models serve` endpoint, and print the results."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--dataset", default="FD001")
    parser.add_argument("--input", default=None, help="Path to a raw CMAPSS-format file; defaults to the dataset's test file.")
    parser.add_argument("--n", type=int, default=5, help="Number of engines to score.")
    args = parser.parse_args()

    input_df, true_rul = load_scoring_batch(dataset=args.dataset, path=args.input, n=args.n)

    # The "dataframe_split" payload format tells the MLflow inference server
    # to reconstruct a pandas DataFrame with these exact column names --
    # required here, since TurbofanRULModel.prepare_input selects feature
    # columns *by name*.
    payload = {
        "dataframe_split": {
            "columns": input_df.columns.tolist(),
            "data": input_df.values.tolist(),
        }
    }

    url = f"http://{args.host}:{args.port}/invocations"
    print(f"POST {url}")
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    predictions = response.json()["predictions"]

    _print_results(input_df, predictions, true_rul)


if __name__ == "__main__":
    main()
