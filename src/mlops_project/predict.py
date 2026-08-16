"""Score engines with a registered model, loaded directly in-process.

This is the simplest way to consume a model from the registry: no serving
infrastructure, just `mlflow.pyfunc.load_model(...)` followed by
`.predict(...)`, the same as calling a method on any other Python object.
Compare with `predict_via_rest.py`, which hits the same model over HTTP
instead, the way a separate application (not written in Python, or not
even on the same machine) would have to.

By default this scores engines from the FD001 *test* set, as a stand-in for
"new production data arriving." Note that this exact test set was already
used by train.py to compute the `test_rmse` metric logged for the run --
so the errors printed here should roughly match that logged metric, not
demonstrate anything new about generalization. For a genuine illustration
of unseen data, pass --input pointing at a different file (any file with
the same 26 raw CMAPSS columns).

Usage:
    uv run predict
    uv run predict --n 10
    uv run predict --version 2
    uv run predict --input data/test_FD003.txt --n 10
"""
import argparse
import os

import mlflow
import pandas as pd

from .data import MODEL_ALIAS, MODEL_NAME, load_scoring_batch


def _model_uri(model_name: str, alias: str | None, version: str | None) -> str:
    if version:
        return f"models:/{model_name}/{version}"
    return f"models:/{model_name}@{alias}"


def _print_results(input_df: pd.DataFrame, predictions, true_rul: pd.Series | None) -> None:
    result = pd.DataFrame({
        "unit_number": input_df["unit_number"],
        "predicted_RUL": pd.Series(predictions).round(1).values,
    })
    if true_rul is not None:
        result["true_RUL"] = true_rul.values
        result["abs_error"] = (result["predicted_RUL"] - result["true_RUL"]).abs().round(1)
    print(result.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--alias", default=MODEL_ALIAS, help="Registry alias to load (ignored if --version is given).")
    parser.add_argument("--version", default=None, help="Exact registered version number, instead of an alias.")
    parser.add_argument("--dataset", default="FD001")
    parser.add_argument("--input", default=None, help="Path to a raw CMAPSS-format file; defaults to the dataset's test file.")
    parser.add_argument("--n", type=int, default=5, help="Number of engines to score.")
    args = parser.parse_args()

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_registry_uri(mlflow_uri)

    model_uri = _model_uri(args.model_name, args.alias, args.version)
    print(f"loading model: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    input_df, true_rul = load_scoring_batch(dataset=args.dataset, path=args.input, n=args.n)
    predictions = model.predict(input_df)

    _print_results(input_df, predictions, true_rul)


if __name__ == "__main__":
    main()
