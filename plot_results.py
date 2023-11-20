import pandas as pd
import matplotlib.pyplot as plt
import os

# function to read a file and plot src recall results


def plot_src_recall():
    df = pd.read_csv(
        # r"C:\Users\sarth\442_DataPoisoning_FL\experiment_results\kmnist_cnn\M_0\3002_results.csv",
        r"C:\cpen442\442_DataPoisoning_FL\experiment_results\kmnist_mlp\M_20\kmnist_mlp_m20_3_results.csv",
        header=None,
    )

    src_class = df[13]
    print(src_class)
    epochs = list(range(1, 201))

    plt.plot(epochs, src_class)
    plt.xlabel("Epochs")
    plt.ylabel("Source Recall")

    plt.savefig(
        # r"C:\Users\sarth\442_DataPoisoning_FL\experiment_results\kmnist_cnn\M_0\plots\exp3_src_recall.jpg"
        r"C:\cpen442\442_DataPoisoning_FL\experiment_results\kmnist_mlp\M_20\kmnist_mlp_m20_3_src_recall.jpg"
    )
    plt.show()



def plot_avg_scr_recall():
    src_dir = "C:\\cpen442\\442_DataPoisoning_FL\\experiment_results\\kmnist_mlp\\M_0"

    csv_files = [f for f in os.listdir(src_dir) if f.endswith('results.csv')]

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
    result_df['Average_Column'] = result_df.mean(axis=1)

    epochs = list(range(1, 201))

    plt.plot(epochs, result_df['Average_Column'])
    plt.xlabel("Epochs")
    plt.ylabel("Source Recall")

    plt.savefig(
        f"{src_dir}\\avg_plot"
    )
    plt.show()


def plot_all_avgs():
    src_dir = "C:\\cpen442\\442_DataPoisoning_FL\\experiment_results\\kmnist_mlp"
    
    sets = ["M_0", "M_10", "M_20"]
    
    epochs = list(range(1, 201))
    
    plt.xlabel("Epochs")
    plt.ylabel("Source Recall")
    
    for p_set in sets:
        sub_dir = f"{src_dir}\\{p_set}\\"
        
        csv_files = [f for f in os.listdir(sub_dir) if f.endswith('results.csv')]

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

        

        plt.plot(epochs, mean, label=p_set)


    plt.legend()
    plt.savefig(
        f"{src_dir}\\avg_plot"
    )
    plt.show()

plot_all_avgs()
# plot_src_recall()
