import mlrun
import pandas as pd
from mlrun.frameworks.sklearn import apply_mlrun
from sklearn import ensemble


@mlrun.handler()
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
):
    # Pick an ideal ML model
    model = ensemble.RandomForestClassifier()

    # Wrap our model with Mlrun features, specify the test dataset for analysis and accuracy measurements
    apply_mlrun(model, model_name="model", x_test=X_test, y_test=y_test)

    # Train our model
    model.fit(X_train, y_train)
