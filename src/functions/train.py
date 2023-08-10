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
    bootstrap: bool,
    max_depth: int,
    min_samples_leaf: int,
    min_samples_split: int,
    n_estimators: int,
):
    # Pick an ideal ML model
    model = ensemble.RandomForestClassifier(
        bootstrap=bootstrap,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        n_estimators=n_estimators,
    )

    # Wrap our model with Mlrun features, specify the test dataset for analysis and accuracy measurements
    apply_mlrun(model, model_name="model", x_test=X_test, y_test=y_test)

    # Train our model
    model.fit(X_train, y_train)
