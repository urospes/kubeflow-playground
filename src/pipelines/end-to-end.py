from kfp import dsl, client
from pipelines.components import extract_data as extractor
from pipelines.components import data_visualization as visualization
from pipelines.components import feature_transformation as feature_transformator
from pipelines.components import train as trainer
from pipelines.components import serving as server


@dsl.pipeline(name="maternity-prediction")
def maternity_prediction_pipeline():
    extract_task = extractor.extract_data()
    visualization.visualize_data(dataset=extract_task.output)
    transformation_task = feature_transformator.feature_transformation(
        raw_dataset=extract_task.output, test_size=0.2
    )
    visualization.visualize_data(dataset=transformation_task.outputs["train_dataset"])
    visualization.visualize_data(dataset=transformation_task.outputs["test_dataset"])
    # with dsl.ParallelFor([1e-2, 1e-3]) as learning_rate:
    train_task = trainer.train(
        train_dataset=transformation_task.outputs["train_dataset"],
        test_dataset=transformation_task.outputs["test_dataset"],
        learning_rate=1e-3,
        n_epochs=3,
    )
    server.serve_model(
        model=train_task.outputs["kfp_model"],
        preprocessor=transformation_task.outputs["transformer"],
    )


if __name__ == "__main__":
    kfp_client = client.Client()
    run = kfp_client.create_run_from_pipeline_func(
        maternity_prediction_pipeline,
    )
