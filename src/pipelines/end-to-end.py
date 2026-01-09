from kfp import dsl, client
from pipelines import components

@dsl.pipeline(name="maternity-prediction")
def maternity_prediction_pipeline():
    extract_task = components.extract_data()


if __name__ == "__main__":
    kfp_client = client.Client()
    run = kfp_client.create_run_from_pipeline_func(
        maternity_prediction_pipeline,
    )
