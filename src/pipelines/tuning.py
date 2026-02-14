from typing import List
from kfp import dsl, client
from pipelines.components import extract_data as extractor
from pipelines.components import data_visualization as visualization
from pipelines.components import feature_transformation as feature_transformator
from pipelines.components import train as trainer


@dsl.pipeline(name="maternity-health-model-tuning")
def model_tuning_pipeline(
    test_size: float,
    layer_configs: List[List[int]],
    learning_rates: List[float],
    n_epochs: List[int],
):
    extract_task = extractor.extract_data()
    visualization.visualize_data(dataset=extract_task.output)
    transformation_task = feature_transformator.feature_transformation(
        raw_dataset=extract_task.output, test_size=test_size
    )
    visualization.visualize_data(dataset=transformation_task.outputs["train_dataset"])
    visualization.visualize_data(dataset=transformation_task.outputs["test_dataset"])

    with dsl.ParallelFor(layer_configs) as layer_config:
        with dsl.ParallelFor(learning_rates) as learning_rate:
            with dsl.ParallelFor(n_epochs) as n_epoch:
                trainer.train(
                    train_dataset=transformation_task.outputs["train_dataset"],
                    test_dataset=transformation_task.outputs["test_dataset"],
                    layer_config=layer_config,
                    learning_rate=learning_rate,
                    n_epochs=n_epoch,
                )


if __name__ == "__main__":
    training_hyperparameter_space = {
        "layer_configs": [[6, 6, 3], [6, 8, 3]],
        "learning_rates": [1e-2, 1e-3],
        "n_epochs": [3],
    }

    kfp_client = client.Client()
    run = kfp_client.create_run_from_pipeline_func(
        model_tuning_pipeline,
        arguments={
            **training_hyperparameter_space,
            "test_size": 0.2,
        },
    )
