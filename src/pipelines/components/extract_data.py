from kfp import dsl


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["polars", "boto3"],
)
def extract_data(raw_dataset: dsl.Output[dsl.Dataset]):
    import polars as pl
    import io
    import boto3
    from botocore.client import Config

    MINIO_ENDPOINT = "http://minio-access-service.kubeflow.svc.cluster.local:9000"
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id="minio",
        aws_secret_access_key="minio123",
        config=Config(signature_version="s3v4"),
    )
    file = s3.get_object(Bucket="iot-data", Key="mat_health.csv")["Body"].read()
    data = pl.read_csv(io.BytesIO(file), infer_schema_length=None)
    print(data.head())
    data.write_csv(raw_dataset.path)