import cloudpickle
import mlrun
import numpy as np
import pandas as pd


class SKLearnPipeline:
    """
    Apply sklearn pipeline to incoming record. Receives a
    dict and returns a dict with the sklearn processing in between.
    """

    def __init__(self, pipeline_path: str):
        local_path = mlrun.get_dataitem(pipeline_path).local()
        with open(local_path, "rb") as f:
            self.pipeline = cloudpickle.load(f)

    def do(self, event: dict) -> dict:
        df_event = pd.DataFrame(event)
        df_transformed = pd.DataFrame(
            data=self.pipeline.transform(df_event),
            columns=self.pipeline.get_feature_names_out(),
        )
        return df_transformed.to_dict(orient="record")


class ClassifierModel(mlrun.serving.V2ModelServer):
    """
    Model server for individual model. Describes load and predict behavior
    """

    def load(self):
        """load and initialize the model and/or other elements"""
        model_file, extra_data = self.get_model(".pkl")
        self.model = cloudpickle.load(open(model_file, "rb"))

    def predict(self, body: dict) -> list:
        """Generate model predictions from sample."""
        feats = np.asarray(body["inputs"])
        result: np.ndarray = self.model.predict(feats)
        return result.tolist()
