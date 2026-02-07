from kfp import dsl


@dsl.component(
    base_image="pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime",
    packages_to_install=["pandas==2.3.3", "kubeflow==0.2.1", "onnx"],
)
def train(
    train_dataset: dsl.Input[dsl.Dataset],
    test_dataset: dsl.Input[dsl.Dataset],
    kfp_model: dsl.Output[dsl.Model],
    learning_rate: float = 1e-3,
    n_epochs: int = 1,
):
    from typing import Tuple
    import pandas as pd
    import torch
    import onnx
    import os

    class PandasDataset(torch.utils.data.Dataset):
        def __init__(self, csv_path: str, target_col: str):
            dataframe = pd.read_csv(csv_path)
            self.features = dataframe.drop(target_col, axis=1)
            self.labels = dataframe[target_col]

        def __len__(self):
            return self.features.shape[0]

        def __getitem__(self, idx: int):
            return (
                torch.tensor(self.features.iloc[idx], dtype=torch.float32),
                torch.tensor(self.labels.iloc[idx], dtype=torch.long),
            )

    class NNClassifier(torch.nn.Module):
        def __init__(self, layer_config: Tuple[Tuple[int, int]]):
            super().__init__()
            self.layers = torch.nn.Sequential()
            for i, (layer_in, layer_out) in enumerate(layer_config):
                self.layers.append(torch.nn.Linear(layer_in, layer_out))
                if i < len(layer_config) - 1:
                    self.layers.append(torch.nn.ReLU())

        def forward(self, x):
            return self.layers(x)

    def train_model(
        model: NNClassifier,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        device: str,
    ):
        model.train()

        train_size = len(dataloader.dataset)
        for batch, (x_train, y_train) in enumerate(dataloader):
            x_train = x_train.to(device)
            y_train = y_train.to(device)

            y_pred = model(x_train)
            loss = loss_fn(y_pred, y_train)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 10 == 0:
                loss, current = loss.item(), (batch + 1) * len(x_train)
                print(f"loss: {loss:>7f}  [{current:>5d}/{train_size:>5d}]")

    def evaluate_model(
        model: NNClassifier,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: str,
    ):
        model.eval()
        test_loss, n_correct = 0, 0
        with torch.no_grad():
            for x_test, y_test in dataloader:
                x_test = x_test.to(device)
                y_test = y_test.to(device)
                y_pred = model(x_test)
                test_loss += loss_fn(y_pred, y_test).item()
                n_correct += (y_pred.argmax(1) == y_test).type(torch.float).sum().item()

        print(
            f"Accuracy: {(100 * n_correct / len(dataloader.dataset)):>0.1f}%, Avg. loss: {test_loss / len(dataloader):>8f} \n"
        )

    def train_loop(
        model: NNClassifier,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        n_epochs: int,
        device: str,
    ):
        for i in range(n_epochs):
            print(f"Epoch {i + 1}\n------------------------------------------")
            train_model(
                model=model,
                dataloader=train_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
            )
            evaluate_model(
                model=model,
                dataloader=test_dataloader,
                loss_fn=loss_fn,
                device=device,
            )
        print("Training finished.")

    device, backend = ("cuda", "nccl") if torch.cuda.is_available() else ("cpu", "gloo")
    # torch.distributed.init_process_group(backend=backend)
    # print(
    #     "Distributed Training with WORLD_SIZE: {}, RANK: {}, LOCAL_RANK: {}.".format(
    #         torch.distributed.get_world_size(),
    #         torch.distributed.get_rank(),
    #         int(os.getenv("LOCAL_RANK", 0)),
    #     )
    # )

    train_data = PandasDataset(csv_path=train_dataset.path, target_col="RiskLevel")
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        # sampler=torch.utils.data.DistributedSampler(train_data, shuffle=True),
    )
    test_data = PandasDataset(csv_path=test_dataset.path, target_col="RiskLevel")
    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=32,
        shuffle=False,
        # sampler=torch.utils.data.DistributedSampler(test_data, shuffle=False),
    )

    # model = torch.nn.parallel.DistributedDataParallel(
    #     NNClassifier(layer_config=((6, 6), (6, 3))).to(device)
    # )
    model = NNClassifier(layer_config=((6, 6), (6, 3))).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loop(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        n_epochs=n_epochs,
        device=device,
    )

    model_dir = os.path.join(kfp_model.path, "maternity-data-model", "1")
    os.makedirs(model_dir, exist_ok=True)
    onnx_path = os.path.join(model_dir, "model.onnx")
    dummy_input = torch.randn(1, 6, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["features"],
        output_names=["risk_prediction"],
        dynamic_axes={
            "features": {0: "batch_size"},
            "risk_prediction": {0: "batch_size"},
        },
        opset_version=17,
    )
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")
    print(onnx.helper.printable_graph(onnx_model.graph))

    # torch.distributed.barrier()
    # if torch.distributed.get_rank() == 0:
    #     torch.save(model.state_dict(), kfp_model.path)
    #     print("Training is finished")
    # torch.distributed.destroy_process_group()
