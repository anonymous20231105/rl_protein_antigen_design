# pip install numpy matplotlib scipy
import random
import time
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from src import utils_read_fasta
from src.utils_esmfold_plddt import esmfold_sequence_2_plddt, create_model
from src.utils_basic import wait_gpu_cool


def ndarray_2_ax(ax, data, title=None, set_x_max=None, set_x_min=None, xlabel='Data'):
    # Calculate the kernel density estimation
    kde = gaussian_kde(data)  # up to 100000
    # Define the range over which to evaluate the density
    xmin, xmax = data.min(), data.max()
    if set_x_max is not None:
        xmax = set_x_max
    if set_x_min is not None:
        xmin = set_x_min
    x = np.linspace(xmin, xmax, xmax-xmin)

    # Evaluate the density on the defined range
    print("density = kde(x)...")
    start_time = time.time()
    density = kde(x)
    print("density = kde(x) time: ", time.time()-start_time)

    ax.plot(x, density)
    ax.fill_between(x, density, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    if title is not None:
        ax.set_title(title)

    return density


def main_length():
    # Generate some random data
    seq_datasets = utils_read_fasta.read_dataset_from_fasta()
    length_list = []
    print("len(seq_datasets): ", len(seq_datasets))
    suit_num = 0
    for i in seq_datasets:
        temp_len = len(i)
        length_list.append(temp_len)
        if 50 <= temp_len <= 150:
            suit_num += 1
    print("suit_num: ", suit_num)
    data = np.array(length_list)

    # Create the density plot
    _, ax = plt.subplots(figsize=(16 * 0.4, 8 * 0.4))

    density = ndarray_2_ax(ax, data, set_x_max=600, title=None, xlabel="Length")
    x_50_150 = np.linspace(50, 150, 150-50)
    ax.fill_between(x_50_150, density[50:150], alpha=0.5)
    ax.plot([50, 50], [0, max(density)], "--", c="orange")
    ax.plot([150, 150], [0, max(density)], "--", c="orange")
    plt.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.15)

    print("plt.show()...")
    plt.savefig("../data/processed/fig_dataset_length_distribute.pdf")
    plt.show()


def main_length_2part():
    # Generate some random data
    seq_datasets = utils_read_fasta.read_dataset_from_fasta()
    print("len(seq_datasets): ", len(seq_datasets))
    seq_datasets_1 = seq_datasets[:int(len(seq_datasets)*0.8)]
    seq_datasets_2 = seq_datasets[int(len(seq_datasets)*0.8):]

    # Create the density plot
    _, ax = plt.subplots(figsize=(16 * 0.4, 8 * 0.4))

    length_list = []
    for i in seq_datasets_1:
        temp_len = len(i)
        length_list.append(temp_len)
    data = np.array(length_list)
    density = ndarray_2_ax(ax, data, set_x_max=1000, title=None, xlabel="Length")

    length_list = []
    for i in seq_datasets_2:
        temp_len = len(i)
        length_list.append(temp_len)
    data = np.array(length_list)
    density = ndarray_2_ax(ax, data, set_x_max=1000, title=None, xlabel="Length")


    plt.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.15)

    print("plt.show()...")
    plt.savefig("../data/processed/fig_dataset_length_distribute_2part.pdf")
    plt.show()


def main_plddt_entire():
    # Generate some random data
    dataset_all = utils_read_fasta.read_dataset_from_fasta()
    print("len(dataset_all): ", len(dataset_all))
    dataset_suit = []
    for i in dataset_all:
        if 50 <= len(i) <= 150:
            dataset_suit.append(i)
    print("len(dataset_suit): ", len(dataset_suit))

    print("esm_model = create_model(chunk_size=128)...")
    esm_model = create_model(chunk_size=128)

    plddt_list = []
    for i in tqdm.tqdm(dataset_suit):
        plddt, _, _, _ = esmfold_sequence_2_plddt(esm_model, i)
        # print("plddt: ", plddt)
        plddt_list.append(plddt)

    np.save("../data/processed/dataset_plddt", plddt_list)


def calc_plddt_and_save(sample_num=10000):
    # Generate some random data
    dataset_all = utils_read_fasta.read_dataset_from_fasta()
    print("len(dataset_all): ", len(dataset_all))
    dataset_suit = []
    for i in dataset_all:
        if 50 <= len(i) <= 150:
            dataset_suit.append(i)
    print("len(dataset_suit): ", len(dataset_suit))
    print("esm_model = create_model(chunk_size=128)...")
    esm_model = create_model(chunk_size=128)
    plddt_list = []
    for i in tqdm.tqdm(range(sample_num)):
        seq_index = random.randint(0, len(dataset_suit))
        seq = dataset_suit[seq_index]
        plddt, _, _, _ = esmfold_sequence_2_plddt(esm_model, seq)
        if plddt is None:
            continue
        # print("plddt: ", plddt)
        plddt_list.append(plddt)
        wait_gpu_cool(58)
    np.save("../data/processed/dataset_plddt_" + str(sample_num), plddt_list)


def load_plddt_and_plot():
    plddt_list = np.load("../data/processed/dataset_plddt_10000.npy", allow_pickle=True)
    mean = np.mean(plddt_list)
    print("mean: ", mean)
    plddt_list = plddt_list.astype(int)
    _, ax = plt.subplots(figsize=(16 * 0.4, 8 * 0.4))
    density = ndarray_2_ax(ax, plddt_list, title=None, xlabel="plddt", set_x_min=20)

    ax.plot([mean, mean], [0-max(density)*0.02, max(density)*1.02], "--", c="green")

    plt.subplots_adjust(left=0.11, right=0.97, top=0.97, bottom=0.15)
    plt.savefig("../data/processed/fig_dataset_plddt_distribute.pdf")
    print("plt.show()...")
    plt.show()


def main_plddt_part():
    # calc_plddt_and_save()
    load_plddt_and_plot()


def main_dataset_rl():
    list_plddt_dataset = np.load("../data/processed/dataset_plddt_10000.npy", allow_pickle=True)
    list_plddt_rl = np.load("../data/processed/dataset_rl_plddt_1000.npy", allow_pickle=True)
    list_plddt_random = np.load("../data/processed/dataset_random_plddt_1000.npy", allow_pickle=True)

    mean_dataset = np.mean(list_plddt_dataset)
    print("mean_dataset: ", mean_dataset)
    mean_rl = np.mean(list_plddt_rl)
    print("mean_rl: ", mean_rl)
    mean_random = np.mean(list_plddt_random)
    print("mean_random: ", mean_random)

    list_plddt_dataset = list_plddt_dataset.astype(int)
    list_plddt_rl = list_plddt_rl.astype(int)
    list_plddt_random = list_plddt_random.astype(int)

    _, ax = plt.subplots(figsize=(16 * 0.4, 8 * 0.4))

    density = ndarray_2_ax(ax, list_plddt_dataset, title=None, xlabel="plddt", set_x_min=20)
    ax.plot([mean_dataset, mean_dataset], [0 - max(density) * 0.02, max(density) * 1.02], "--", c="C0", alpha=0.5, label="Dataset")

    density = ndarray_2_ax(ax, list_plddt_rl, title=None, xlabel="plddt", set_x_min=20)
    ax.plot([mean_rl, mean_rl], [0 - max(density) * 0.02, max(density) * 1.02], "--", c="C1", alpha=0.5, label="RL")

    density = ndarray_2_ax(ax, list_plddt_random, title=None, xlabel="plddt", set_x_min=20)
    ax.plot([mean_random, mean_random], [0 - max(density) * 0.02, max(density) * 1.02], "--", c="C2", alpha=0.5, label="MC")

    plt.legend()

    plt.subplots_adjust(left=0.11, right=0.97, top=0.97, bottom=0.15)
    plt.savefig("../data/processed/fig_dataset_plddt_distribute_dataset_rl_random.pdf")
    print("plt.show()...")
    plt.show()


if __name__ == "__main__":
    # main_length()
    main_length_2part()
    # main_plddt_entire()
    # main_plddt_part()
    # main_dataset_rl()
