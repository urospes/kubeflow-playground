from kfp import dsl


@dsl.component(
    base_image="pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime",
    packages_to_install=["pandas"],
)
def train(
    train_dataset: dsl.Input[dsl.Dataset],
    test_dataset: dsl.Input[dsl.Dataset],
    model: dsl.Output[dsl.Model],
):
    def build_model():
        pass

    def build_data_loader():
        pass

    def train_model():
        pass
