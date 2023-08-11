import mlrun
from kfp import dsl


@dsl.pipeline(name="GitOps Training Pipeline", description="Train a model")
def pipeline(
    existing_model_path: str = "None",
    force_deploy: bool = False,
    source_url: str = "https://s3.wasabisys.com/iguazio/data/model-monitoring/iris_dataset.csv",
    label_column: str = "label",
    post_github: bool = True,
):
    # Get our project object
    project = mlrun.get_current_project()

    # Ingest data
    ingest = project.run_function(
        "data",
        handler="get_data",
        inputs={"data": source_url},
        outputs=["data"],
    )
    
    # Validate data integrity
    validate_data_integrity = project.run_function(
        "validate",
        handler="validate_data_integrity",
        inputs={"data": ingest.outputs["data"]},
        params={"label_column": label_column},
        outputs=["passed_suite"],
    )
    
    # Analyze data
    project.run_function(
        "describe",
        inputs={"table": ingest.outputs["data"]},
        params={"label_column": label_column},
    )
    
    # Process data
    process = project.run_function(
        "data",
        handler="process_data",
        inputs={"data": ingest.outputs["data"]},
        params={"label_column": label_column, "test_size": 0.10},
        outputs=["train", "test"],
    ).after(validate_data_integrity)
    
    # Validate train test split
    validate_train_test_split = project.run_function(
        "validate",
        handler="validate_train_test_split",
        inputs={"train" : process.outputs["train"], "test" : process.outputs["test"]},
        params={"label_column": label_column},
        outputs=["passed_suite"]
    )

    
#     with dsl.Condition(
#         validate_data_integrity.outputs["passed_suite"] == False or 
#         validate_train_test_split.outputs["passed_suite"] == False
#     ):
#         project.run_function("fail", params={"message" : "Data validation failed"})
    
#     with dsl.Condition(validate_data_integrity.outputs["passed_suite"] == True and 
#         validate_train_test_split.outputs["passed_suite"] == True
#     ):
    train = project.run_function(
        "train",
        inputs={
            "train": process.outputs["train"],
            "test": process.outputs["test"],
        },
        params={"label_column":label_column},
        hyperparams={
            "bootstrap": [True, False],
            "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            "min_samples_leaf": [1, 2, 4],
            "min_samples_split": [2, 5, 10],
            "n_estimators": [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
        },
        selector="max.accuracy",
        hyper_param_options=mlrun.model.HyperParamOptions(
            strategy="random", max_iterations=5
        ),
        outputs=["model"],
    ).after(validate_train_test_split)

    validate_model = project.run_function(
        "validate",
        handler="validate_model",
        inputs={
            "train": process.outputs["train"],
            "test": process.outputs["test"],
        },
        params={
            "model_path" : train.outputs["model"],
            "label_column" : label_column
        },
        outputs=["passed_suite"]
    )

    # Deploy model to endpoint
    serving_fn = project.get_function("serving")
    serving_fn.set_tracking()
    deploy = project.deploy_function(
        serving_fn, models=[{"key": "model", "model_path": train.outputs["model"]}]
    ).after(validate_model)
