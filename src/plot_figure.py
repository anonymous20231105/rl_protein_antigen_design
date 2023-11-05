import random

import numpy as np
import matplotlib.pyplot as plt

from src.utils_plot import cut_list, draw_time_aver




def read_one_file(dt_str, cut_steps=None):
    file_name = "../data/interim/tensorboard_his/" + dt_str + "/training_curve" + dt_str + ".npy"
    list_data = np.load(file_name, allow_pickle=True).tolist()
    list_train_return, list_train_steps, list_valid_return, list_valid_steps = list_data
    assert len(list_train_return) == len(list_train_steps)
    assert len(list_valid_return) == len(list_valid_steps)
    if cut_steps:
        list_train_return, list_train_steps = cut_list(cut_steps, list_train_return, list_train_steps)
        list_valid_return, list_valid_steps = cut_list(cut_steps, list_valid_return, list_valid_steps)
    return list_train_return, list_train_steps, list_valid_return, list_valid_steps


def create_fig(figsize):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    return ax


def plot_figure_training_curve():
    # config
    dt_str_1 = "20230326_201714"
    dt_str_2 = "20230327_102637"  # "20230326_211825"
    dt_str_3 = "20230326_221551"
    FIG_SIZE_11 = (16 * 0.3, 16 * 0.3)

    print("Reading...")
    list_train_return_1, list_train_steps_1, list_valid_return_1, list_valid_steps_1 = read_one_file(dt_str_1,
                                                                                                     cut_steps=200000)
    list_train_return_2, list_train_steps_2, list_valid_return_2, list_valid_steps_2 = read_one_file(dt_str_2,
                                                                                                     cut_steps=200000)
    list_train_return_3, list_train_steps_3, list_valid_return_3, list_valid_steps_3 = read_one_file(dt_str_3,
                                                                                                     cut_steps=200000)

    print("Processing...")
    ax = create_fig(figsize=FIG_SIZE_11)
    draw_time_aver([list_train_steps_1, list_train_steps_2, list_train_steps_3],
                   [list_train_return_1, list_train_return_2, list_train_return_3],
                   ax, step=200, name="train", color="b")
    draw_time_aver([list_valid_steps_1, list_valid_steps_2, list_valid_steps_3],
                   [list_valid_return_1, list_valid_return_2, list_valid_return_3],
                   ax, step=50, name="valid", color="y")
    plt.xlabel("Steps")
    plt.ylabel("Returns")
    plt.legend()

    print("Saving...")
    plt.savefig("../data/processed/fig_plot.pdf")


def add_legend_seq_length():
    seq_length = 50
    seq_length_norm = (seq_length - 50) / 100
    plt.scatter(0, 0, color=(1, seq_length_norm, 1 - seq_length_norm, 0.5), s=10, label="seq_length: 50")
    seq_length = 100
    seq_length_norm = (seq_length - 50) / 100
    plt.scatter(0, 0, color=(1, seq_length_norm, 1 - seq_length_norm, 0.5), s=10, label="seq_length: 100")
    seq_length = 150
    seq_length_norm = (seq_length - 50) / 100
    plt.scatter(0, 0, color=(1, seq_length_norm, 1 - seq_length_norm, 0.5), s=10, label="seq_length: 150")
    plt.legend()


def plot_figure_entire_specify():
    FIG_SIZE_11 = (16 * 0.25, 16 * 0.25)

    print("Reading...")
    file_name = "../data/processed/test_result.npy"
    result_list = np.load(file_name, allow_pickle=True).tolist()

    print("Processing...")
    ax = create_fig(figsize=FIG_SIZE_11)
    # print("result_list: ", result_list)
    plt.xlim([65, 86])  # 30
    plt.ylim([37, 91])  # 70
    plt.xlabel("plddt entire")
    plt.ylabel("plddt specify")
    plt.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.11)
    for result in result_list:
        plddt_entire = result[0]
        plddt_specify = result[1]
        seq_length = result[2]
        print("seq_length: ", seq_length)
        seq_length_norm = (seq_length - 50) / 100
        # seq_length_norm = seq_length_norm * 0.5 # + 0.5
        plt.scatter(plddt_entire, plddt_specify, color=(1, seq_length_norm, 1-seq_length_norm, 0.5), s=10)
    add_legend_seq_length()
    print("Saving...")
    plt.savefig("../data/processed/fig_plot_entire_specify.pdf")


def add_one_plddt_legend(plddt_entire, plddt_mean, plddt_std):
    plddt_entire_norm = (plddt_entire - plddt_mean) / plddt_std / 3 + 0.5
    plddt_entire_norm = np.clip(plddt_entire_norm, a_max=1, a_min=0)
    plt.scatter(0, 0, color=(1-plddt_entire_norm, plddt_entire_norm, 1, 0.5), s=10,
                label="plddt: " + str(plddt_entire))


def add_legend_plddt(plddt_mean, plddt_std):
    for plddt_entire in [90, 80, 70, 60, 50]:
        add_one_plddt_legend(plddt_entire, plddt_mean, plddt_std)
    plt.legend()


def plot_figure_length():
    FIG_SIZE_11 = (16 * 0.25, 16 * 0.25)

    print("Reading...")
    file_name = "../data/processed/test_result_length.npy"  # _230406
    result_array = np.load(file_name, allow_pickle=True)
    plddt_array = result_array[:, 2]
    plddt_mean = np.mean(plddt_array)
    plddt_std = np.std(plddt_array)
    print("plddt_mean: ", plddt_mean)
    print("plddt_std: ", plddt_std)
    result_list = result_array.tolist()

    print("Processing...")
    ax = create_fig(figsize=FIG_SIZE_11)
    plt.xlim([50-2, 150+2])
    plt.ylim([50*0.1-2, 150*0.9+2])
    plt.xlabel("sequence length")
    plt.ylabel("design length")
    plt.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.11)
    for result in result_list:
        seq_length = result[0]
        pred_length = result[1]
        plddt_entire = result[2]

        # print("np.mean(plddt_entire): ", np.mean(plddt_entire))
        # print("np.std(plddt_entire): ", np.std(plddt_entire))

        # plddt_entire_norm = plddt_entire / 100
        plddt_entire_norm = (plddt_entire - plddt_mean) / plddt_std / 3 + 0.5
        plddt_entire_norm = np.clip(plddt_entire_norm, a_max=1, a_min=0)
        plt.scatter(seq_length, pred_length, color=(1-plddt_entire_norm, plddt_entire_norm, 1, 0.5), s=5)
    add_legend_plddt(plddt_mean, plddt_std)
    print("Saving...")
    plt.savefig("../data/processed/fig_plot_length.pdf")


def plot_53_ddg():
    FIG_SIZE_11 = (16 * 0.25, 16 * 0.25)

    print("Reading...")
    file_name = "../data/processed/ppo_test_53_ddg_result.npy"  # _230406
    result_array = np.load(file_name, allow_pickle=True)
    result_list = result_array.tolist()

    ddg_array = result_array[:, 2]
    ddg_mean = np.mean(ddg_array)
    ddg_std = np.std(ddg_array)

    print("Processing...")
    ax = create_fig(figsize=FIG_SIZE_11)
    plt.xlim([5, 25])
    plt.ylim([45-5, 80])
    plt.xlabel("cdrh3 length (amino acid num)")
    plt.ylabel("plddt stability (%)")
    plt.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.11)
    for result in result_list:
        [ddrh3_len, plddt_cdrh3, ddg] = result
        ddg_norm = (ddg - ddg_mean) / ddg_std / 3 + 0.5
        ddg_norm = np.clip(ddg_norm, a_max=1, a_min=0)
        # ddrh3_len += (random.random() - 0.5) * 2
        plt.scatter(ddrh3_len, plddt_cdrh3, color=(1-ddg_norm, 0, ddg_norm, 0.5), s=15)

    for ddg in [-1, -3, -5, -7, -9]:
        ddg_norm = (ddg - ddg_mean) / ddg_std / 3 + 0.5
        ddg_norm = np.clip(ddg_norm, a_max=1, a_min=0)
        plt.scatter(0, 0, color=(1-ddg_norm, 0, ddg_norm, 0.5), s=10,
                    label="ddg: " + str(ddg))
    plt.legend()

    print("Saving...")
    plt.savefig("../data/processed/fig_53_ddg.pdf")


def plot_53_ddg_diff():
    FIG_SIZE_11 = (16 * 0.25, 16 * 0.25)

    print("Reading...")
    file_name_rl = "../data/processed/ppo_test_53_ddg_result.npy"  # _230406
    file_name_random = "../data/processed/result_random_1.npy"  # ppo_test_53_ddg_result_random
    file_name_mean = "../data/processed/result_mean_1.npy"
    result_array_rl = np.load(file_name_rl, allow_pickle=True)
    result_array_random = np.load(file_name_random, allow_pickle=True)
    result_array_mean = np.load(file_name_mean, allow_pickle=True)
    ddg_array_rl = result_array_rl[:, 2]
    print("np.mean(ddg_array_rl): ", np.mean(ddg_array_rl))
    ddg_array_random = result_array_random[:, 2]
    print("np.mean(ddg_array_random): ", np.mean(ddg_array_random))
    ddg_array_mean = result_array_mean[:, 2]
    print("np.mean(ddg_array_mean): ", np.mean(ddg_array_mean))
    result_list_rl = result_array_rl.tolist()
    result_list_random = result_array_random.tolist()
    result_list_mean = result_array_mean.tolist()

    print("Processing...")
    ax = create_fig(figsize=FIG_SIZE_11)
    plt.xlim([40, 85])
    plt.ylim([-10, 5])
    # plt.xlabel("cdrh3 length (amino acid num)")
    plt.xlabel("plddt stability (%)")
    plt.ylabel("affinity (ddg)")
    plt.subplots_adjust(left=0.16, right=0.97, top=0.97, bottom=0.13)
    for result in result_list_random:
        [ddrh3_len, plddt_cdrh3, ddg] = result
        plt.scatter(plddt_cdrh3, ddg, color=(0, 1, 0.5, 0.5), s=20)
    for result in result_list_mean:
        [ddrh3_len, plddt_cdrh3, ddg] = result
        plt.scatter(plddt_cdrh3, ddg, color=(0, 0.5, 1, 0.5), s=20)
    for result in result_list_rl:
        [ddrh3_len, plddt_cdrh3, ddg] = result
        plt.scatter(plddt_cdrh3, ddg, color=(1, 0, 0.5, 0.5), s=20)

    plt.scatter(0, 0, color=(0, 1, 0.5, 0.5), s=20, label="MC")
    plt.scatter(0, 0, color=(0, 0.5, 1, 0.5), s=20, label="MEAN")
    plt.scatter(0, 0, color=(1, 0, 0.5, 0.5), s=20, label="Ours")
    plt.legend()

    print("Saving...")
    plt.savefig("../data/processed/fig_53_ddg_diff.pdf")


def main():
    # plot_figure_training_curve()
    # plot_figure_entire_specify()
    # plot_figure_length()
    # plot_53_ddg()
    plot_53_ddg_diff()

    print("Finished...")
    plt.pause(100)


if __name__ == '__main__':
    main()
