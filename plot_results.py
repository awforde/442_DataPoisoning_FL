import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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

# plot_avg_src_recall(
    # r"C:\Users\sarth\442_DataPoisoning_FL\experiment_results\artbench_cnn\M_0",
    # r"artbench_cnn_m0",
    # False,
# )


def plot_avg_scr_recall():
    src_dir = "C:\\cpen442\\442_DataPoisoning_FL\\experiment_results\\kmnist_mlp\\M_0"

    csv_files = [f for f in os.listdir(src_dir) if f.endswith("results.csv")]

    # Initialize an empty dataframe to store the averages
    result_df = pd.DataFrame()

    # Loop through each CSV file
    for csv_file in csv_files:
        # Construct the full path to the CSV file
        file_path = os.path.join(src_dir, csv_file)

        # Read the CSV file
        df = pd.read_csv(file_path, header=None)

        # Extract the first column
        col = df.iloc[:, 13]

        # Add the first column to the result dataframe
        result_df[csv_file] = col

    # Calculate the average of all columns
    result_df["Average_Column"] = result_df.mean(axis=1)

    epochs = list(range(1, 201))

    plt.plot(epochs, result_df["Average_Column"])
    plt.xlabel("Epochs")
    plt.ylabel("Source Recall")

    plt.savefig(f"{src_dir}\\avg_plot")
    plt.show()


def plot_all_avgs():
    experiment_name = "artbench_cnn"
    src_dir = f"C:\\cpen442\\442_DataPoisoning_FL\\experiment_results\\{experiment_name}"

    sets = ["M_0", "M_10", "M_20", "M_40"]

    epochs = list(range(1, 201))

    plt.xlabel("Epochs")
    plt.ylabel("Source Recall")
    plt.title(experiment_name)

    for p_set in sets:
        sub_dir = f"{src_dir}\\{p_set}\\"

        csv_files = [f for f in os.listdir(sub_dir) if f.endswith("results.csv")]

        # print(csv_files)

        # Initialize an empty dataframe to store the averages
        result_df = pd.DataFrame()

        # Loop through each CSV file
        for csv_file in csv_files:
            # Construct the full path to the CSV file
            file_path = os.path.join(sub_dir, csv_file)

            # Read the CSV file
            df = pd.read_csv(file_path, header=None)

            # Extract the first column
            col = df.iloc[:, 13]

            # Add the first column to the result dataframe
            result_df[csv_file] = col

        # Calculate the average of all columns
        mean = result_df.mean(axis=1)

        plt.plot(epochs, mean, label=f"m={p_set.split('_')[-1]}%")

    plt.legend()
    plt.savefig(f"{src_dir}\\avg_plot")
    plt.show()


# Plots the mean source recall of all models and all datasets for different M
# This isused to show how source recall decreases as M increases
def plot_all_exps():
    src_dir = "C:\\cpen442\\442_DataPoisoning_FL\\experiment_results"

    sets = ["M_0", "M_10", "M_20", "M_40"]

    epochs = list(range(1, 201))
    plt.xlabel("Epochs")
    plt.ylabel("Source Recall")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    title = "Avg Accuracy of All Models vs Poisoned Workers"
    plt.title(title)

    experiments = [f for f in os.listdir(src_dir) if (f.endswith("mlp") or f.endswith("cnn")) and "kmnist" in f]

    for m_set in sets:

        # Initialize an empty dataframe to store the averages
        result_df = pd.DataFrame()

        for experiment in experiments:
            sub_dir = f"{src_dir}\\{experiment}\\{m_set}\\"

            csv_files = [f for f in os.listdir(sub_dir) if f.endswith("results.csv")]

            # Loop through each CSV file
            for csv_file in csv_files:
                # Construct the full path to the CSV file
                file_path = os.path.join(sub_dir, csv_file)

                # Read the CSV file
                df = pd.read_csv(file_path, header=None)

                # Extract the first column
                col = df.iloc[:, 0]

                # Add the first column to the result dataframe
                result_df[csv_file] = col

        # Calculate the average of all columns
        mean = result_df.mean(axis=1)

        plt.plot(epochs, mean, label=f"m={m_set.split('_')[-1]}%")

    plt.legend()
    # plt.savefig(f"{src_dir}\\src_recall_all_exps")
    plt.savefig(f"C:\\cpen442\\442_DataPoisoning_FL\\plots\\{title}.png")
    plt.show()


# Show the difference between how CNNs and MLPs respond to poisoning
#
# Because MLPs and CNNs achieve different performance, we should show the degradation relative to M_0
def plot_mlp_vs_cnn():

    src_dir = "C:\\cpen442\\442_DataPoisoning_FL\\experiment_results"

    sets = ["M_10", "M_20", "M_40"]

    epochs = list(range(1, 201))
    plt.xlabel("Epochs")
    plt.ylabel("Source Recall difference from m=0%")
    title = "MNIST CNN vs MLP response to poisoning"
    plt.title(title, loc='left')
    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    # ax.set_ylim([None, 150])
    plot_colours = {
        "M_10" : "blue",
        "M_20" : "orange",
        "M_40" : "green"
    }

    # MNIST Only
    experiments = [f for f in os.listdir(src_dir) if f.endswith("mnist_mlp") or f.endswith("mnist_cnn")]

    # CIFAR Only
    # experiments = [f for f in os.listdir(src_dir) if (f.endswith("mlp") or f.endswith("cnn")) and not("mnist" in f)]

    # Both Set Styles
    # experiments = [f for f in os.listdir(src_dir) if f.endswith("mlp") or f.endswith("cnn")]


    # Initialize an empty dataframe to store the averages
    m0_df = pd.DataFrame()

    for experiment in experiments:
        sub_dir = f"{src_dir}\\{experiment}\\M_0\\"

        csv_files = [f for f in os.listdir(sub_dir) if f.endswith("results.csv")]

        result_df = pd.DataFrame()

        # Loop through each CSV file
        for csv_file in csv_files:
            # Construct the full path to the CSV file
            file_path = os.path.join(sub_dir, csv_file)

            # Read the CSV file
            df = pd.read_csv(file_path, header=None)

            # Extract the first column
            col = df.iloc[:, 13]

            # Add the first column to the result dataframe
            result_df[csv_file] = col

        # Calculate the average of all columns
        m0_df[experiment] = result_df.mean(axis=1)

    for m_set in sets:
        mlp_df = pd.DataFrame()
        cnn_df = pd.DataFrame()

        for experiment in experiments:
            sub_dir = f"{src_dir}\\{experiment}\\{m_set}\\"

            csv_files = [f for f in os.listdir(sub_dir) if f.endswith("results.csv")]

            # Initialize an empty dataframe to store the averages
            result_df = pd.DataFrame()

            # Loop through each CSV file
            for csv_file in csv_files:
                # Construct the full path to the CSV file
                file_path = os.path.join(sub_dir, csv_file)

                # Read the CSV file
                df = pd.read_csv(file_path, header=None)

                # Extract the first column
                col = df.iloc[:, 13]

                # Add the first column to the result dataframe
                result_df[csv_file] = col

            # Calculate the average of all columns
            mean = result_df.mean(axis=1)

            diff = abs(mean - m0_df[experiment])/m0_df[experiment]

            if experiment.endswith("mlp"):
                mlp_df[experiment] = diff
            else:
                cnn_df[experiment] = diff

        mlp = mlp_df.mean(axis=1) * 100

        plt.plot(epochs, mlp, label=f"mlp, m={m_set.split('_')[-1]}%", linestyle="solid", color=plot_colours[m_set])

        cnn = cnn_df.mean(axis=1) * 100

        plt.plot(epochs, cnn, label=f"cnn, m={m_set.split('_')[-1]}%", linestyle="dotted", color=plot_colours[m_set])


    plt.legend(loc='upper right', bbox_to_anchor=(1.12, 1.1))
    # plt.savefig(f"{src_dir}\\{title}")
    plt.savefig(f"C:\\cpen442\\442_DataPoisoning_FL\\plots\\{title}.png")
    plt.show()




# Show the difference between how datasets respond to poisoning
#
# Because datasets achieve different performance, we should show the degradation relative to M_0
def plot_dataset_response():

    src_dir = "C:\\cpen442\\442_DataPoisoning_FL\\experiment_results"

    # sets = ["M_10", "M_20", "M_40"]
    sets = ["M_10"]

    epochs = list(range(1, 201))
    plt.xlabel("Epochs")
    plt.ylabel("Source Recall difference from m=0%")
    title = "Dataset Response to Poisoning, m=10%"
    plt.title(title)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plot_colours = {
        "M_10" : "blue",
        "M_20" : "orange",
        "M_40" : "green"
    }


    # experiments = [f for f in os.listdir(src_dir) if (f.endswith("mlp") or f.endswith("cnn")) and not("artbench" in f)]
    # experiments = [f for f in os.listdir(src_dir) if f.endswith("mlp") or f.endswith("cnn")]
    experiments = [f for f in os.listdir(src_dir) if f.endswith("cnn")]

    # Initialize an empty dataframe to store the averages
    m0_df = pd.DataFrame()

    for experiment in experiments:
        sub_dir = f"{src_dir}\\{experiment}\\M_0\\"

        csv_files = [f for f in os.listdir(sub_dir) if f.endswith("results.csv")]

        result_df = pd.DataFrame()

        # Loop through each CSV file
        for csv_file in csv_files:
            # Construct the full path to the CSV file
            file_path = os.path.join(sub_dir, csv_file)

            # Read the CSV file
            df = pd.read_csv(file_path, header=None)

            # Extract the first column
            col = df.iloc[:, 13]

            # Add the first column to the result dataframe
            result_df[csv_file] = col

        # Calculate the average of all columns
        m0_df[experiment] = result_df.mean(axis=1)

    for m_set in sets:

        for experiment in experiments:
            sub_dir = f"{src_dir}\\{experiment}\\{m_set}\\"

            csv_files = [os.path.join(sub_dir, f) for f in os.listdir(sub_dir) if f.endswith("results.csv")]

            # Initialize an empty dataframe to store the averages
            result_df = pd.DataFrame()

            # Loop through each CSV file
            for csv_file in csv_files:
                # Read the CSV file
                df = pd.read_csv(csv_file, header=None)

                # Extract the first column
                col = df.iloc[:, 13]

                # Add the first column to the result dataframe
                result_df[csv_file] = col

            # Calculate the average of all columns
            mean = result_df.mean(axis=1)

            diff = abs(mean - m0_df[experiment])/m0_df[experiment]


            plt.plot(epochs, diff * 100, label=experiment.split('_')[0])

    # plt.legend(loc='upper right', bbox_to_anchor=(1.12, 1.1))
    plt.legend()
    # plt.savefig(f"C:\\cpen442\\442_DataPoisoning_FL\\plots\\{title}.png")
    plt.show()




# Show the difference between how datasets respond to poisoning
#
# Because datasets achieve different performance, we should show the degradation relative to M_0
def plot_dataset_response_both_models():

    src_dir = "C:\\cpen442\\442_DataPoisoning_FL\\experiment_results"

    # sets = ["M_10", "M_20", "M_40"]
    sets = ["M_10"]

    epochs = list(range(1, 201))
    plt.xlabel("Epochs")
    plt.ylabel("Source Recall difference from m=0%")
    title = "Dataset Response to Poisoning, m=10%"
    plt.title(title)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plot_colours = {
        "M_10" : "blue",
        "M_20" : "orange",
        "M_40" : "green"
    }


    # experiments = [f for f in os.listdir(src_dir) if (f.endswith("mlp") or f.endswith("cnn")) and not("artbench" in f)]
    # experiments = [f for f in os.listdir(src_dir) if f.endswith("mlp") or f.endswith("cnn")]
    experiments = [f for f in os.listdir(src_dir) if f.endswith("cnn")]

    # Initialize an empty dataframe to store the averages
    m0_df = pd.DataFrame()

    for experiment in experiments:
        sub_dir = f"{src_dir}\\{experiment}\\M_0\\"

        csv_files = [os.path.join(sub_dir, f) for f in os.listdir(sub_dir) if f.endswith("results.csv")]

        sub_dir2 = f"{src_dir}\\{experiment.split('_')[0]}_mlp\\M_0\\"

        csv_files += [os.path.join(sub_dir2, f) for f in os.listdir(sub_dir2) if f.endswith("results.csv")]


        result_df = pd.DataFrame()

        # Loop through each CSV file
        for csv_file in csv_files:

            # Read the CSV file
            df = pd.read_csv(csv_file, header=None)

            # Extract the first column
            col = df.iloc[:, 13]

            # Add the first column to the result dataframe
            result_df[csv_file] = col

        # Calculate the average of all columns
        m0_df[experiment] = result_df.mean(axis=1)
        # print(experiment)

    for m_set in sets:

        for experiment in experiments:
            sub_dir = f"{src_dir}\\{experiment}\\{m_set}\\"

            csv_files = [os.path.join(sub_dir, f) for f in os.listdir(sub_dir) if f.endswith("results.csv")]

            sub_dir2 = f"{src_dir}\\{experiment.split('_')[0]}_mlp\\{m_set}\\"

            csv_files += [os.path.join(sub_dir2, f) for f in os.listdir(sub_dir2) if f.endswith("results.csv")]

            # Initialize an empty dataframe to store the averages
            result_df = pd.DataFrame()

            # Loop through each CSV file
            for csv_file in csv_files:
                # Construct the full path to the CSV file
                # file_path = os.path.join(sub_dir, csv_file)

                # Read the CSV file
                df = pd.read_csv(csv_file, header=None)

                # Extract the first column
                col = df.iloc[:, 13]

                # Add the first column to the result dataframe
                result_df[csv_file] = col

            # Calculate the average of all columns
            mean = result_df.mean(axis=1)

            diff = abs(mean - m0_df[experiment])/m0_df[experiment]

            plt.plot(epochs, diff * 100, label=experiment.split('_')[0])

    # plt.legend(loc='upper right', bbox_to_anchor=(1.12, 1.1))
    plt.legend()
    plt.savefig(f"C:\\cpen442\\442_DataPoisoning_FL\\plots\\{title}.png")
    plt.show()



# Plot the source recall for each dataset at M_0
# This shows the relative difficulty of each dataset
def plot_dataset_difficulty():
    src_dir = "C:\\cpen442\\442_DataPoisoning_FL\\experiment_results"

    epochs = list(range(1, 201))
    plt.xlabel("Epochs")
    plt.ylabel("Source Recall")
    m_set = "M_20"
    title = f"Baseline Source Recall across Datasets, {m_set}"
    plt.title(title)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())


    experiments = [f for f in os.listdir(src_dir) if f.endswith("cnn")]

    for experiment in experiments:
        sub_dir = f"{src_dir}\\{experiment}\\{m_set}\\"

        csv_files = [os.path.join(sub_dir, f) for f in os.listdir(sub_dir) if f.endswith("results.csv")]

        sub_dir2 = f"{src_dir}\\{experiment.split('_')[0]}_mlp\\{m_set}\\"

        csv_files += [os.path.join(sub_dir2, f) for f in os.listdir(sub_dir2) if f.endswith("results.csv")]


        result_df = pd.DataFrame()

        # Loop through each CSV file
        for csv_file in csv_files:
            # Read the CSV file
            df = pd.read_csv(csv_file, header=None)

            # Extract the first column
            col = df.iloc[:, 13]

            # Add the first column to the result dataframe
            result_df[csv_file] = col


        plt.plot(epochs, result_df.mean(axis=1) * 100, label=experiment.split('_')[0])



    # plt.legend(loc='upper right', bbox_to_anchor=(1.12, 1.1))
    plt.legend()
    plt.savefig(f"C:\\cpen442\\442_DataPoisoning_FL\\plots\\{title}.png")
    plt.show()




# plot_all_exps()

# plot_mlp_vs_cnn()
# plot_dataset_response_both_models()
plot_all_exps()
# plot_dataset_difficulty()
# plot_all_avgs()
# plot_src_recall()
