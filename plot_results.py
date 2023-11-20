import pandas as pd
import matplotlib.pyplot as plt
import os

# function to read a file and plot src recall results


# Plot source recall for i_th experiment
def plot_src_recall(parentFolder, db_model_percent):
    i = 3
    df = pd.read_csv(
        os.path.join(parentFolder, db_model_percent + f"_{i}_results.csv"),
        header=None,
    )

    src_class = df[13]

    epochs = list(range(1, 201))

    plt.plot(epochs, src_class)
    plt.xlabel("Epochs")
    plt.ylabel("Source Recall")
    # plt.ylabel("Accuracy")
    plt.savefig(
        os.path.join(parentFolder, r"plots", db_model_percent + f"_{i}_srcrecall.jpg")
    )
    plt.show()


# plot average source recall across all 3 experiments
def plot_avg_src_recall(parentFolder, db_model_percent, recall):
    src_recalls = 0

    for i in range(1, 4, 1):
        df = pd.read_csv(
            os.path.join(parentFolder, db_model_percent + f"_{i}_results.csv"),
            header=None,
        )
        if recall:
            src_recalls = src_recalls + df[13]
        else:
            src_recalls = src_recalls + df[0]

    avg_src_recalls = src_recalls / 3

    epochs = list(range(1, 201))

    plt.plot(epochs, avg_src_recalls)
    plt.xlabel("Epochs")

    if recall:
        plt.ylabel("Average Source Recall")
        plt.savefig(
            os.path.join(
                parentFolder, r"plots", db_model_percent + r"_avg_srcrecall.jpg"
            )
        )
    else:
        plt.ylabel("Avg Accuracy")
        plt.savefig(
            os.path.join(parentFolder, r"plots", db_model_percent + r"_avg_acc.jpg")
        )

    plt.show()


# plot_src_recall(
#     r"C:\Users\sarth\442_DataPoisoning_FL\experiment_results\artbench_cnn\M_0",
#     r"artbench_cnn_m0",
# )

plot_avg_src_recall(
    r"C:\Users\sarth\442_DataPoisoning_FL\experiment_results\artbench_cnn\M_0",
    r"artbench_cnn_m0",
    False,
)
