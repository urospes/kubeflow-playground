from kfp import dsl


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["kserve", "scikit-learn"],
)
def serve_model(model: dsl.Input[dsl.Model], preprocessor: dsl.Input[dsl.Artifact]):
    from kubernetes import client
    from kserve import (
        constants,
        KServeClient,
        V1beta1InferenceService,
        V1beta1InferenceServiceSpec,
        V1beta1PredictorSpec,
        V1beta1TransformerSpec,
        V1beta1TritonSpec,
    )

    name = "maternity-data-model"
    namespace = "kubeflow-user-example-com"

    if model.path.startswith("/minio/"):
        model_path = "s3://" + model.path[len("/minio/") :]

    if preprocessor.path.startswith("/minio/"):
        preprocessor_path = "s3://" + preprocessor.path[len("/minio/") :]

    transformer_spec = V1beta1TransformerSpec(
        min_replicas=1,
        containers=[
            client.V1Container(
                name="inference-preprocessor",
                image="urospes/inference-preprocessor-custom:latest",
                args=[
                    "--model_name",
                    name,
                    "--transformer_uri",
                    preprocessor_path,
    
                ],
                env=[
                    client.V1EnvVar(
                        name="AWS_ACCESS_KEY_ID",
                        value_from=client.V1EnvVarSource(
                            secret_key_ref=client.V1SecretKeySelector(
                                name="minio-kserve-secret",
                                key="AWS_ACCESS_KEY_ID",
                            )
                        ),
                    ),
                    client.V1EnvVar(
                        name="AWS_SECRET_ACCESS_KEY",
                        value_from=client.V1EnvVarSource(
                            secret_key_ref=client.V1SecretKeySelector(
                                name="minio-kserve-secret",
                                key="AWS_SECRET_ACCESS_KEY",
                            )
                        ),
                    ),
                    client.V1EnvVar(
                        name="AWS_ENDPOINT_URL",
                        value="http://minio-service.kubeflow:9000",
                    ),
                    client.V1EnvVar(
                        name="S3_USE_HTTPS",
                        value="0",
                    ),
                    client.V1EnvVar(
                        name="S3_VERIFY_SSL",
                        value="0",
                    ),
                ],
            )
        ],
        service_account_name="kserve-minio-sa",
    )
    predictor = V1beta1PredictorSpec(
        min_replicas=1,
        onnx=V1beta1TritonSpec(
            storage_uri=model_path,
        ),
        service_account_name="kserve-minio-sa",
    )

    isvc = V1beta1InferenceService(
        api_version=constants.KSERVE_V1BETA1,
        kind="InferenceService",
        metadata=client.V1ObjectMeta(
            name=name,
            namespace=namespace,
            annotations={"sidecar.istio.io/inject": "false"},
        ),
        spec=V1beta1InferenceServiceSpec(
            predictor=predictor, transformer=transformer_spec
        ),
    )

    kserve_client = KServeClient()
    kserve_client.create(isvc)

    kserve_client.wait_isvc_ready(name, namespace=namespace)
