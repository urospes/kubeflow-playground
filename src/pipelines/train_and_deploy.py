from typing import List
from kfp import dsl, client
from pipelines.components import extract_data as extractor
from pipelines.components import data_visualization as visualization
from pipelines.components import feature_transformation as feature_transformator
from pipelines.components import train_distributed as trainer
from pipelines.components import serving as serving


@dsl.pipeline(name="maternity-model-training")
def train_and_deploy_pipeline(
    test_size: float, layer_config: List[int], learning_rate: float, n_epochs: int
):
    extract_task = extractor.extract_data()
    visualization.visualize_data(dataset=extract_task.output)
    transformation_task = feature_transformator.feature_transformation(
        raw_dataset=extract_task.output, test_size=test_size
    )
    visualization.visualize_data(dataset=transformation_task.outputs["train_dataset"])
    visualization.visualize_data(dataset=transformation_task.outputs["test_dataset"])

    train_task = trainer.train(
        train_dataset=transformation_task.outputs["train_dataset"],
        test_dataset=transformation_task.outputs["test_dataset"],
        layer_config=layer_config,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
    )
    train_task.set_caching_options(False)

    # # TODO: modify this task to serve model from registry (or from minio bucket)
    # serving_task = serving.serve_model(
    #     model=train_task.outputs["kfp_model"],
    #     preprocessor=transformation_task.outputs["transformer"],
    # )
    # serving_task.set_caching_options(False)


if __name__ == "__main__":
    kfp_client = client.Client()
    run = kfp_client.create_run_from_pipeline_func(
        train_and_deploy_pipeline,
        arguments={
            "layer_config": [6, 6, 3],
            "learning_rate": 1e-3,
            "n_epochs": 3,
            "test_size": 0.2,
        },
    )
