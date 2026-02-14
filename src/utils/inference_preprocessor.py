import argparse
import os
import logging
import cloudpickle
import kserve
import numpy as np
import pandas as pd
from kserve_storage import Storage
from kserve import InferInput, InferRequest
from kserve.model import PredictorConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MaternityTransformer(kserve.Model):
    def __init__(
        self, name: str, predictor_config: PredictorConfig, transformer_uri: str
    ):
        super().__init__(name, predictor_config)
        self.transformer_uri = transformer_uri
        self.load()

    def load(self):
        self.col_transformer = self.__load_preprocessor(path=self.transformer_uri)
        self.ready = True

    @staticmethod
    def __load_preprocessor(path: str):
        local_path = Storage.download(path)
        if os.path.isdir(local_path):
            files = os.listdir(local_path)
            local_path = os.path.join(local_path, files[0])
        with open(local_path, "rb") as f:
            preprocessor = cloudpickle.load(f)
        return preprocessor

    def preprocess(self, payload: dict, headers: dict) -> dict:
        transformed = self.col_transformer.transform(
            pd.DataFrame(payload["features"])
        ).values.astype(np.float32)
        logger.info("RAW: %s", payload)
        logger.info("TRANSFORMED %s", transformed)
        infer_input = InferInput(
            name="features",
            shape=list(transformed.shape),
            datatype="FP32",
            data=transformed.flatten().tolist(),
        )

        return InferRequest(
            model_name=self.name,
            infer_inputs=[infer_input],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
    parser.add_argument(
        "--transformer_uri",
        required=True,
        help="Storage URI for the cloudpickle'd transformer artifact",
    )
    args, _ = parser.parse_known_args()
    predictor_config = PredictorConfig(
        predictor_host=args.predictor_host,
        predictor_protocol="v2",  # args.predictor_protocol,
        predictor_use_ssl=args.predictor_use_ssl,
        predictor_request_timeout_seconds=args.predictor_request_timeout_seconds,
        predictor_request_retries=args.predictor_request_retries,
        predictor_health_check=args.enable_predictor_health_check,
    )
    model = MaternityTransformer(
        name=args.model_name,
        predictor_config=predictor_config,
        transformer_uri=args.transformer_uri,
    )
    kserve.ModelServer(predictor_config=predictor_config).start(models=[model])
