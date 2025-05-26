import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_losses(
    datasets,
    x_col,
    y_cols,
    labels,
    styles=None,
    title="Loss Comparison",
    xlabel="Epoch",
    ylabel="Loss",
    log_scale=False,
    legend_loc='upper right'
):
    """
    Trace une ou plusieurs courbes de pertes à partir de plusieurs datasets.
    datasets: liste de DataFrames ou dicts pandas
    x_col: nom de la colonne pour l'axe x (str)
    y_cols: liste des noms de colonnes à tracer (str)
    labels: liste des labels pour chaque courbe
    styles: liste de dicts pour les styles matplotlib (facultatif)
    title, xlabel, ylabel: titres et labels
    log_scale: bool, si True utilise une échelle log pour y
    ylim_zero: bool, si True force y>=0
    legend_loc: position de la légende
    """
    sns.set_style("darkgrid")
    plt.figure(figsize=(12, 8))
    if styles is None:
        styles = [{} for _ in range(len(y_cols))]
    for i, (data, y_col, label, style) in enumerate(zip(datasets, y_cols, labels, styles)):
        plt.plot(
            data[x_col], data[y_col],
            label=label,
            linewidth=2.5,
            marker='o',
            markersize=6,
            **style
        )
    plt.xlabel(xlabel, fontsize=14, fontweight='bold')
    plt.ylabel(ylabel + (" (log scale)" if log_scale else ""), fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    if log_scale:
        plt.yscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(fontsize=12, frameon=True, shadow=True, borderpad=1, loc=legend_loc)
    plt.tight_layout()
    plt.show()

data_train = pd.read_csv('train_losses_task1_basic.csv')
data_test = pd.read_csv('test_losses_task1_basic.csv')
data_train_reversed = pd.read_csv('train_losses_task2_reversed.csv')
data_test_reversed = pd.read_csv('test_losses_task2_reversed.csv')
data_train_reversed_no_dropout = pd.read_csv('train_losses_task4_reversed_no_dropout.csv')
data_test_reversed_no_dropout = pd.read_csv('test_losses_task4_reversed_no_dropout.csv')
data_train_no_dropout = pd.read_csv('train_losses_task3_no_dropout.csv')
data_test_no_dropout = pd.read_csv('test_losses_task3_no_dropout.csv')

plot_losses(
    datasets=[
        data_test,
        data_test_reversed,
        data_test_reversed_no_dropout,
        data_test_no_dropout
    ],
    x_col="epoch",
    y_cols=["test_loss", "test_loss", "test_loss", "test_loss"],
    labels=[
        "Test Basic",
        "Test Reversed",
        "Test Reversed No Dropout",
        "Test No Dropout"
    ],
    title="Test Loss Comparison",
    xlabel="Epoch",
    ylabel="Loss",
    log_scale=True,
    legend_loc='upper right',
    styles=[
        {"color": "blue", "linestyle": "-"},
        {"color": "orange", "linestyle": "--"},
        {"color": "green", "linestyle": "-."},
        {"color": "red", "linestyle": ":"}
    ]
)