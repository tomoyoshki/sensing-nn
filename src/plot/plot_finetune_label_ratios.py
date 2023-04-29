import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
import matplotlib
from matplotlib.colors import ListedColormap

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

RATIOS = ["100% Labels", "10% Labels", "1% Labels"]
N_Tasks = len(RATIOS)


def plot_group_bar(
    data_matrix,
    dataset_name,
    metrics,
    labels,
    out_dir,
    muls=[-5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
    bar_width=0.07,
    y_limit=(0.2, 0.9),
    seaborn_font_scale=1.7,
    seaborn_style="whitegrid",
):
    N_metrics = len(metrics)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    sns.set(font_scale=seaborn_font_scale)
    sns.set_style(seaborn_style)
    sns.set_palette(sns.color_palette("Paired", 12))

    x = np.arange(N_Tasks)
    fig, ax_tuples = plt.subplots(1, N_metrics, figsize=(10, 3.5), dpi=80, sharey=False)

    for metric_idx, metric in enumerate(metrics):
        ax = ax_tuples[metric_idx] if N_metrics > 1 else ax_tuples
        # ax.set_xlabel()
        ax.set_ylabel(f"{metric}")
        ax.set_xticks(x)
        ax.set_xticklabels(RATIOS)
        ax.set_ylim(y_limit)
        for label_idx, label in enumerate(labels):
            data = np.squeeze(data_matrix[metric][label])

            # draw bar
            ax.bar(x + muls[label_idx] * bar_width, data, bar_width, label=label)

    ax = ax_tuples[-1] if N_metrics > 1 else ax_tuples
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, bbox_to_anchor=(0.48, 0.88), loc="lower center", ncol=6, handlelength=0.7, columnspacing=0.7
    )
    output_path = os.path.join(out_dir, f"./{dataset_name}_finetune_label_ratios.pdf")
    plt.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    out_dir = "/home/sl29/FoundationSense/result/figures"
    metrics = ["Accuracy"]
    labels = ["SimCLR", "MoCo", "CMC", "MAE", "Cosmo", "Cocoa", "MTSS", "TS2Vec", "GMC", "TNC", "TS-TCC", "FOCAL"]

    # ------------------------ RealWorld_HAR ------------------------
    data_matrix = {
        "Accuracy": {
            "SimCLR": [0.9250, 0.8891, 0.7523],
            "MoCo": [0.9390, 0.9073, 0.7482],
            "CMC": [0.9129, 0.8691, 0.6994],
            "MAE": [0.7803, 0.6561, 0.3764],
            "Cosmo": [0.3429, 0.2122, 0.1753],
            "Cocoa": [0.7040, 0.6869, 0.6122],
            "MTSS": [0.4206, 0.3799, 0.3113],
            "TS2Vec": [0.7254, 0.6522, 0.4750],
            "GMC": [0.8640, 0.7712, 0.5191],
            "TNC": [0.8533, 0.8436, 0.7996],
            "TS-TCC": [0.8734, 0.8564, 0.7473],
            "FOCAL": [0.9772, 0.9593, 0.8840],
        },
    }
    plot_group_bar(data_matrix, "Parkland", metrics, labels, out_dir, y_limit=(0.2, 1.0))
