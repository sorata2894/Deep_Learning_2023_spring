import torch
import numpy as np
from torch.utils.data.dataset import Dataset

class transform_dataset(Dataset):
    def __init__(self, signal, label):
        self.X_data = torch.from_numpy(np.asarray(signal, dtype=np.float32)).float()
        self.Y_Data = torch.from_numpy(np.asarray(label, dtype=np.float32)).float()

    def __getitem__(self, index):
        x = self.X_data[index]
        y = self.Y_Data[index]

        entry = {'data': x, 'label': y}
    
        return entry

    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.X_data)
    
def plot_comparison_result(epochs, ELU_test, ReLU_test, LeakyReLU_test, ELU_train, ReLU_train, LeakyReLU_train, Net_name = "EEG"):
    import matplotlib.pyplot as plt
    save_fig_name = Net_name + "Net_AUC_comparison.png"
    fig_title = "Activation function comparison(" + Net_name + "Net)"
    epochs_list = [i+1 for i in range(epochs)]
    fig , ax = plt.subplots()
    plt.plot(epochs_list, ELU_test, label="ELU_test")
    plt.plot(epochs_list, ReLU_test, label="ReLU_test")
    plt.plot(epochs_list, LeakyReLU_test, label="LeakyReLU_test")
    plt.plot(epochs_list, ELU_train, label="ELU_train")
    plt.plot(epochs_list, ReLU_train, label="ReLU_train")
    plt.plot(epochs_list, LeakyReLU_train, label="LeakyReLU_train")
    plt.title(fig_title)
    plt.xlabel("epochs")
    plt.ylabel("Accuracy(%)")
    ax.legend(loc="lower right")
    plt.savefig(save_fig_name)