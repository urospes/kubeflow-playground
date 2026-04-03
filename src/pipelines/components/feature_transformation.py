from kfp import dsl


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["pandas", "scikit-learn", "cloudpickle"],
)
def feature_transformation(
    raw_dataset: dsl.Input[dsl.Dataset],
    train_dataset: dsl.Output[dsl.Dataset],
    test_dataset: dsl.Output[dsl.Dataset],
    transformer: dsl.Output[dsl.Artifact],
    test_size: float = 0.2,
):
    import pandas as pd
    import sklearn
    from sklearn.metrics.pairwise import rbf_kernel
    import numpy as np
    import cloudpickle

    def rbf_similarity(
        data: pd.DataFrame, mode: int | float, gamma: float
    ) -> np.ndarray:
        return rbf_kernel(
            np.asarray(data).reshape(-1, 1), np.array(mode).reshape(-1, 1), gamma=gamma
        )

    def power_transform(data: np.ndarray | pd.Series, power: float):
        return np.pow(data, power)

    with open(raw_dataset.path) as dataset_file:
        dataset = pd.read_csv(dataset_file)
    print(dataset.head())

    label_col = ["vehicle_fuel"]
    numeric_no_skew_cols = ["vehicle_noise"]
    numeric_left_skew_cols = []
    numeric_right_skew_cols = [
        "vehicle_CO",
        "vehicle_CO2",
        "vehicle_HC",
        "vehicle_NOx",
        "vehicle_PMx",
    ]
    numeric_slight_right_skew_cols = ["vehicle_speed"]

    preprocessor = sklearn.compose.ColumnTransformer(
        transformers=[
            (
                "numeric_no_skew",
                sklearn.preprocessing.RobustScaler(),
                numeric_no_skew_cols,
            ),
            (
                "numeric_right_skew",
                sklearn.preprocessing.FunctionTransformer(
                    np.log1p, feature_names_out="one-to-one"
                ),
                numeric_right_skew_cols,
            ),
            (
                "numeric_slight_right_skew",
                sklearn.preprocessing.FunctionTransformer(
                    np.sqrt, feature_names_out="one-to-one"
                ),
                numeric_slight_right_skew_cols,
            ),
            (
                "numeric_left_skew",
                sklearn.pipeline.Pipeline(
                    [
                        (
                            "power_transform",
                            sklearn.preprocessing.FunctionTransformer(
                                power_transform,
                                kw_args={"power": 2},
                                feature_names_out="one-to-one",
                            ),
                        ),
                        ("normalization", sklearn.preprocessing.MinMaxScaler()),
                    ]
                ),
                numeric_left_skew_cols,
            ),
            (
                "multimodal_vehicle_angle",
                sklearn.preprocessing.FunctionTransformer(
                    rbf_similarity,
                    kw_args={"mode": 250, "gamma": 0.005},
                    feature_names_out="one-to-one",
                ),
                ["vehicle_angle"],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    label_transformer = sklearn.preprocessing.FunctionTransformer(
        np.log1p, feature_names_out="one-to-one"
    ).set_output(transform="pandas")

    train, test = sklearn.model_selection.train_test_split(
        dataset, test_size=test_size, random_state=42
    )

    x_train = preprocessor.fit_transform(train.drop(columns=label_col))
    y_train = label_transformer.fit_transform(train[label_col])
    train = pd.concat([x_train, y_train], axis=1)
    print(train.info())

    x_test = preprocessor.transform(test.drop(columns=label_col))
    y_test = label_transformer.transform(test[label_col])
    test = pd.concat([x_test, y_test], axis=1)
    print(test.info())

    train.to_csv(train_dataset.path, index=False)
    test.to_csv(test_dataset.path, index=False)
    with open(transformer.path, "wb") as f:
        cloudpickle.dump(preprocessor, f)
