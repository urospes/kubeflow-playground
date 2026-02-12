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
    def __init__(self, name: str, predictor_host: str, transformer_uri: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        self.col_transformer = self.__load_preprocessor(path=transformer_uri)
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

    # def postprocess(self, response: dict, headers: dict = None) -> dict:
    #     # Optional: map ordinal predictions back to labels
    #     risk_levels = ["low risk", "mid risk", "high risk"]
    #     predictions = response.get("predictions", [])
    #     labels = [risk_levels[int(round(p))] for p in predictions]
    #     return {"predictions": labels}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
    parser.add_argument(
        "--transformer_uri",
        required=True,
        help="Storage URI for the cloudpickle'd transformer artifact",
    )
    args, _ = parser.parse_known_args()
    logger.info("PARSED ARGS: %s", args)
    model = MaternityTransformer(
        name=args.model_name,
        predictor_host=args.predictor_host,
        transformer_uri=args.transformer_uri,
    )
    # TODO: clean this up, override load method in MaternityTransformer
    # predictor_config = PredictorConfig(
    #     predictor_host=args.predictor_host,
    #     predictor_protocol="v2",
    # )
    model.load()
    # model.predictor_config = predictor_config
    kserve.ModelServer().start(models=[model])
