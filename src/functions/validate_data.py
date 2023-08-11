import mlrun
import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity


@mlrun.handler(outputs=["passed_suite", "suite_report:file"])
def validate_data_integrity(full_data: pd.DataFrame, label_column: str):
    # Get numerical and categorical columns based on dataframe types
    numerical_columns = list(
        set(full_data.select_dtypes("number").columns) - set([label_column])
    )
    categorical_columns = list(
        set(full_data.select_dtypes("object").columns) - set([label_column])
    )
    
    # Create deepchecks dataset with column metadata
    dataset = Dataset(
        df=full_data, label=label_column, cat_features=categorical_columns
    )

    # Run suite
    data_integrity_suite = data_integrity()
    suite_result = data_integrity_suite.run(dataset)

    # Export results
    passed_suite = suite_result.passed()
    suite_report = suite_result.save_as_html()
    
    return passed_suite, suite_report
