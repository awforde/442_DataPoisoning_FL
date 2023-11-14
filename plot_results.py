import pandas as pd
import matplotlib.pyplot as plt

# function to read a file and plot src recall results


def plot_src_recall():
    df = pd.read_csv(
        r"C:\Users\sarth\442_DataPoisoning_FL\experiment_results\kmnist_cnn\M_20\3002_results.csv",
        header=None,
    )

    src_class = df[13]
    print(src_class)
    epochs = list(range(1, 201))

    plt.plot(epochs, src_class)
    plt.xlabel("Epochs")
    plt.ylabel("Source Recall")
    plt.show()
    plt.savefig(
        r"C:\Users\sarth\442_DataPoisoning_FL\experiment_results\kmnist_cnn\M_20\plots\exp3_src_recall.jpg"
    )


plot_src_recall()
