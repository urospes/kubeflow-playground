from kfp import dsl


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["pandas", "scikit-learn"],
)
def feature_transformation(
    raw_dataset: dsl.Input[dsl.Dataset],
    train_dataset: dsl.Output[dsl.Dataset],
    test_dataset: dsl.Output[dsl.Dataset],
    test_size: float = 0.2,
):
    import pandas as pd
    import sklearn
    from sklearn.metrics.pairwise import rbf_kernel
    import numpy as np
    from typing import List

    def transform_labels(label_col: pd.Series) -> pd.Series:
        def label_encoder(label: str) -> int:
            match label:
                case "low risk":
                    return 0
                case "mid risk":
                    return 1
                case "high risk":
                    return 2
                case _:
                    return -1

        return label_col.apply(label_encoder)

    # def rbf_similarity_transform(
    #     dataset: pd.DataFrame, modes: List[float | int]
    # ) -> pd.DataFrame:
    #     return rbf_kernel(dataset, [modes], gamma=0.1)

    with open(raw_dataset.path) as dataset_file:
        dataset = pd.read_csv(dataset_file)
    print(dataset.head())

    label_col = "RiskLevel"
    scale_standard_col_names = ["Age", "SystolicBP", "DiastolicBP"]
    log_transform_col_names = ["BS", "BodyTemp", "HeartRate"]
    # rbf_transform_col_names = ["SystolicBP"]
    scaler = sklearn.preprocessing.StandardScaler()

    train, test = sklearn.model_selection.train_test_split(
        dataset, test_size=test_size, random_state=42, stratify=dataset[label_col]
    )

    scaler = scaler.fit(train[scale_standard_col_names])

    for dataset in (train, test):
        dataset[label_col] = transform_labels(dataset[label_col])
        dataset[scale_standard_col_names] = scaler.transform(
            dataset[scale_standard_col_names]
        )
        dataset[log_transform_col_names] = dataset[log_transform_col_names].transform(
            np.log
        )
        # dataset[rbf_transform_col_names] = rbf_similarity_transform(
        #     dataset=dataset[rbf_transform_col_names], modes=[120]
        # )

    print(train.head())
    print(test.head())

    train.to_csv(train_dataset.path)
    test.to_csv(test_dataset.path)
