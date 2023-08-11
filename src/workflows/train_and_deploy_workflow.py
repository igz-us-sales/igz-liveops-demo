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
    ingest_fn = project.get_function("get-data")
    ingest = project.run_function(
        ingest_fn,
        inputs={"data": source_url},
        params={"label_column": label_column, "test_size": 0.10},
        outputs=["X_train", "X_test", "y_train", "y_test", "full_data"],
    )
    
    # Validate data
    validate_fn = project.get_function("validate-data")
    validate = project.run_function(
        validate_fn,
        inputs={"full_data": ingest.outputs["full_data"]},
        params={"label_column": label_column},
        outputs=["passed_suite"]
    )

    # Analyze data
    describe_fn = project.get_function("describe")
    project.run_function(
        describe_fn,
        inputs={"table": ingest.outputs["full_data"]},
        params={"label_column": label_column},
    )
    
    with dsl.Condition(validate.outputs["passed_suite"] == False):
        project.run_function("fail", params={"message" : "Data validation failed"})
    
    with dsl.Condition(validate.outputs["passed_suite"] == True):
        train_fn = project.get_function("train")
        train = project.run_function(
            train_fn,
            inputs={
                "X_train": ingest.outputs["X_train"],
                "X_test": ingest.outputs["X_test"],
                "y_train": ingest.outputs["y_train"],
                "y_test": ingest.outputs["y_test"],
            },
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
            outputs=["model", "test_set"],
        ).after(validate)

        # Deploy model to endpoint
        serving_fn = project.get_function("serving")
        serving_fn.set_tracking()
        deploy = project.deploy_function(
            serving_fn, models=[{"key": "model", "model_path": train.outputs["model"]}]
        )

    
        
    

    # # Evaluate model and optionally trigger deployment pipeline
    # test_fn = project.get_function("test")
    # project.run_function(
    #     test_fn,
    #     inputs={"test_set": train.outputs["test_set"]},
    #     params={
    #         "label_column": label_column,
    #         "new_model_path": train.outputs["model"],
    #         "existing_model_path": existing_model_path,
    #         "comparison_metric": "accuracy",
    #         "post_github": post_github,
    #         "force_deploy": force_deploy,
    #     },
    # )
