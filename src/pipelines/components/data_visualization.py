from kfp import dsl


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["pandas", "seaborn"],
)
def visualize_data(
    dataset: dsl.Input[dsl.Dataset], visualization: dsl.Output[dsl.HTML]
):
    import seaborn as sb
    import pandas as pd
    import matplotlib.pyplot as plt
    import functools
    import base64
    import io
    from typing import Tuple

    def save_plot(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            fig = func(*args, **kwargs)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            plot_img64 = base64.b64encode(buf.read()).decode("utf-8")
            buf.close()
            plt.close(fig)
            return plot_img64

        return wrapper

    def configure_for_plotting():
        sb.set_style(style="whitegrid")

    @save_plot
    def plot_hist(dataset: pd.DataFrame, cols_of_interest: Tuple[str]) -> plt.Figure:
        fig, ax = plt.subplots(
            1, len(cols_of_interest), figsize=(5 * len(cols_of_interest), 5)
        )
        for i, column in enumerate(cols_of_interest):
            ax[i].set_title(f"{column} histogram")
            sb.histplot(data=dataset, x=column, kde=True, ax=ax[i])
        return fig

    @save_plot
    def plot_corr_matrix(
        dataset: pd.DataFrame, cols_of_interest: Tuple[str]
    ) -> plt.Figure:
        fig, ax = plt.subplots()
        plt.title("Correlation Matrix")
        sb.heatmap(
            dataset[cols_of_interest].corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            square=True,
            ax=ax,
        )
        return fig

    @save_plot
    def pairplot(dataset: pd.DataFrame, cols_of_interest: Tuple[str]) -> plt.Figure:
        grid = sb.pairplot(dataset[cols_of_interest])
        grid.fig.suptitle("Pairplot")
        return grid.fig

    def make_html(plots):
        content = (
            "<!doctype html><html><body><h1>Maternity Health Data Visualizations</h1>"
        )
        for plot in plots:
            content += f'</br><img src="data:image/png;base64, {plot}"></br>'
        content += "</body></html>"
        return content

    # TODO: this is the main component code - refactor inner functions once I switch to
    # Containerized Python KFP components instead of simple Python components
    with open(dataset.path) as dataset_file:
        data = pd.read_csv(dataset_file)

    configure_for_plotting()

    plots = []
    for plot_func in (plot_corr_matrix, pairplot, plot_hist):
        plots.append(
            plot_func(
                dataset=data,
                cols_of_interest=data.select_dtypes(include="number").columns,
            )
        )

    with open(visualization.path, "w") as file:
        file.write(make_html(plots=plots))
