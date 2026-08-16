import os

import mlflow
import mlflow.pyfunc

from .data import MODEL_ALIAS, MODEL_NAME
from .model import DummyModel

def train():
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_registry_uri(mlflow_uri)

    mlflow.set_experiment("dummy_model_experiment")  # TODO: set your experiment name here

    with mlflow.start_run() as run:
        # TODO: here we instantiate a single model and train it. You can alternatively implement a hyperparameter search (e.g. with Optuna -- `uv add optuna`) and train multiple models, logging each one to the MLFlow experiment tracker, but only registering the best one to the model registry.

        model = DummyModel()
        artifacts = model.train(hyperparameters=None, training_input=None) # TODO: pass hyperparameters and training data here

        # You can log parameters and metrics to the MLFlow experiment tracker
        mlflow.log_param("model_type", "dummy_pyfunc")
        mlflow.log_metric("dummy_metric", 0.0)

        # this logs the model to the MLFlow experiment tracker
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=DummyModel(),
            artifacts=artifacts
        )

        model_uri = f"runs:/{run.info.run_id}/model"
        # this registers the model to the MLFlow model registry, under the
        # name predict.py / predict_via_rest.py already expect (see data.py)
        registered_model = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

        # Point the "champion" alias at whichever version was just
        # registered, so predict.py / predict_via_rest.py can always load
        # "the current model" by name, without needing to know version
        # numbers. Every run promotes itself unconditionally here -- a
        # real deployment pipeline would compare against the current
        # champion's metrics first, and only move the alias if better.
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(
            name=MODEL_NAME, alias=MODEL_ALIAS, version=registered_model.version
        )

    return registered_model
