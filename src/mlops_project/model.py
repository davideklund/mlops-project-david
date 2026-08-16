import mlflow
import mlflow.pyfunc

class DummyModel(mlflow.pyfunc.PythonModel):
    def prepare_input(self, model_input):
        return model_input                          # TODO: implement data preparation logic here, to be used both for training and inference.

    def train(self, hyperparameters, training_input):
        training_input = self.prepare_input(training_input)
        model = build_model(hyperparameters)        # TODO: implement model building logic here
        model.train()                               # TODO: implement model training logic here
        # TODO: store your model artifacts, which you will need for inference, on disk. Return a dictionary with the paths to the artifacts.
        # Here's an example using joblib, assuming that `model` is a scikit-learn model. You can use any other serialization method as needed.
        import joblib
        model_path = "sklearn_model.pkl"
        joblib.dump(model, model_path)
        return {'model': model_path} 

    # This is called by MLFlow when loading your model for inference
    # Here you should load your model artifacts based on how you stored them in the `train` method.
    # Here we continue with the joblib + sklearn example.
    def load_context(self, context):
        import joblib
        model_path = context.artifacts["model"]
        self.model = joblib.load(model_path)

    # This is called by MLFlow when making predictions with your model.
    # Here you should implement the logic to make predictions using your model.
    def predict(self, context, model_input, params = None):
        model_input = self.prepare_input(model_input)
        return self.model.predict(model_input)
