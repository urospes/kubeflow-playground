from kfp import dsl


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["pandas", "boto3"],
)
def extract_data(raw_dataset: dsl.Output[dsl.Dataset], file_name: str):
    import pandas as pd
    import io
    import boto3
    from botocore.client import Config

    MINIO_ENDPOINT = "http://minio-access-service.minio.svc.cluster.local:9000"
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
        config=Config(signature_version="s3v4"),
    )
    file = s3.get_object(Bucket="iot-data", Key=file_name)["Body"].read()
    data = pd.read_csv(io.BytesIO(file), sep=";")
    print(data.head())
    data.to_csv(raw_dataset.path, index=False)
