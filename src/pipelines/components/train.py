from typing import List
from kfp import dsl


@dsl.component(
    base_image="pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime",
    packages_to_install=["pandas==2.3.3", "kubeflow==0.2.1"],
)
def train(
    train_dataset: dsl.Input[dsl.Dataset],
    test_dataset: dsl.Input[dsl.Dataset],
    kfp_model: dsl.Output[dsl.Model],
    metrics: dsl.Output[dsl.Metrics],
    layer_config: List[int],
    learning_rate: float = 1e-3,
    n_epochs: int = 1,
):
    from typing import List
    import pandas as pd
    import torch

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
                torch.tensor(self.labels.iloc[idx], dtype=torch.float32).view(1),
            )

    class NNRegressor(torch.nn.Module):
        def __init__(self, layer_config: List[int]):
            super().__init__()
            self.layers = torch.nn.Sequential()
            for i in range(len(layer_config) - 1):
                self.layers.append(
                    torch.nn.Linear(layer_config[i], layer_config[i + 1])
                )
                if i < len(layer_config) - 2:
                    self.layers.append(torch.nn.ReLU())

        def forward(self, x):
            return self.layers(x)

    def train_model(
        model: NNRegressor,
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
        model: NNRegressor,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        device: str,
    ) -> float:
        model.eval()
        test_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for x_test, y_test in dataloader:
                x_test = x_test.to(device)
                y_test = y_test.to(device)
                y_pred = model(x_test)
                test_loss += loss_fn(y_pred, y_test).item()
                all_preds.append(y_pred.cpu())
                all_targets.append(y_test.cpu())

        print(f"Avg. loss: {test_loss / len(dataloader):>8f} \n")
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        mse = torch.nn.functional.mse_loss(all_preds, all_targets).item()
        return mse

    def train_loop(
        model: NNRegressor,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        n_epochs: int,
        device: str,
    ) -> float:
        for i in range(n_epochs):
            print(f"Epoch {i + 1}\n------------------------------------------")
            train_model(
                model=model,
                dataloader=train_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
            )
            eval_metrics = evaluate_model(
                model=model,
                dataloader=test_dataloader,
                loss_fn=loss_fn,
                device=device,
            )
        print("Training finished.")
        return eval_metrics

    device, backend = ("cuda", "nccl") if torch.cuda.is_available() else ("cpu", "gloo")

    train_data = PandasDataset(csv_path=train_dataset.path, target_col="vehicle_fuel")
    print("TRAINING DATASET SIZE:", len(train_data))
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
    )
    test_data = PandasDataset(csv_path=test_dataset.path, target_col="vehicle_fuel")
    print("TEST DATASET SIZE:", len(test_data))
    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=32,
        shuffle=False,
    )

    model = NNRegressor(layer_config=tuple(layer_config)).to(device)
    print(model)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    eval_metrics = train_loop(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        n_epochs=n_epochs,
        device=device,
    )

    metrics.log_metric("MSE", eval_metrics)
