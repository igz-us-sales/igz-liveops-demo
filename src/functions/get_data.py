import mlrun
import pandas as pd
from sklearn.model_selection import train_test_split


@mlrun.handler(outputs=["X_train", "X_test", "y_train", "y_test", "full_data"])
def get_data(
    data: pd.DataFrame, label_column: str, test_size: float, random_state: int = 42
):
    X = data.drop(label_column, axis=1)
    y = data[label_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, data
