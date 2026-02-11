from kfp import dsl


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["kserve", "cloudpickle", "scikit-learn"],
)
def serve_model(model: dsl.Input[dsl.Model], preprocessor: dsl.Input[dsl.Artifact]):
    from kubernetes import client
    from kserve import (
        constants,
        KServeClient,
        V1beta1InferenceService,
        V1beta1InferenceServiceSpec,
        V1beta1PredictorSpec,
        V1beta1TritonSpec,
    )
    import cloudpickle

    name = "maternity-data-model"
    namespace = "kubeflow-user-example-com"

    if model.path.startswith("/minio/"):
        model_path = "s3://" + model.path[len("/minio/") :]

    with open(preprocessor.path, "rb") as f:
        col_transformer = cloudpickle.load(f)

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
        spec=V1beta1InferenceServiceSpec(predictor=predictor),
    )

    kserve_client = KServeClient()
    kserve_client.create(isvc)

    kserve_client.wait_isvc_ready(name, namespace=namespace)
