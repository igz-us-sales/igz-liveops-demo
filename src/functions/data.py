import mlrun
import pandas as pd
from sklearn.model_selection import train_test_split


@mlrun.handler(outputs=["data"])
def get_data(data: pd.DataFrame):
    return data

@mlrun.handler(outputs=["train", "test"])
def process_data(
    data: pd.DataFrame, label_column: str, test_size: float, random_state: int = 42
):
    train, test = train_test_split(
        data, test_size=test_size, random_state=random_state
    )
    return train, test





# get data
# validate data integrity
# proecss data
# validate train test split
# train
# validate model
# deploy