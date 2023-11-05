import os
import random
import re

AC_TABLE = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'K', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
            'W', 'X', 'Y', 'Z']


def list_files_with_extensions(temp_dir, extensions):
    return [f for f in os.listdir(temp_dir) if f.endswith(extensions)]


def parse_fasta(data):
    data = re.sub('>$', '', data, flags=re.M)
    lines = [
                l.replace('\n', '')
                for prot in data.split('>') for l in prot.strip().split('\n', 1)
            ][1:]
    tags, seqs = lines[::2], lines[1::2]

    tags = [t.split()[0] for t in tags]

    return tags, seqs


def read_dataset_from_fasta():
    # "/home/PJLAB/houjing/Study/spinningup/spinup/envs/fasta"
    # "/home/gemhou/Study/src/spinup/envs/fasta"
    fasta_dir = "/home/gemhou/Study/data"
    seqs = None
    for fasta_file in list_files_with_extensions(fasta_dir, (".fasta", ".fa")):
        with open(os.path.join(fasta_dir, fasta_file), "r") as fp:
            data = fp.read()
        tags, seqs = parse_fasta(data)
        # print("tags: ", tags)
    seq_datasets = seqs
    return seq_datasets


def process_data(seq_datasets):
    data_num = len(seq_datasets)
    # print("original data_num: ", data_num)
    seq_datasets = list(set(seq_datasets))
    random.shuffle(seq_datasets)
    data_num = len(seq_datasets)
    # print("data_num: ", data_num)
    return seq_datasets


def calc_reward_factor(seq_datasets):
    count_list = [0] * len(AC_TABLE)
    for data_i in range(len(seq_datasets)):
        for count_j in range(len(count_list)):
            count_list[count_j] += seq_datasets[data_i].count(AC_TABLE[count_j])
    for i in range(len(count_list)):
        if count_list[i] == 0:
            count_list[i] = 1
    percent_list = [i / sum(count_list) for i in count_list]
    standard_percent_list = [1 / len(AC_TABLE)] * len(AC_TABLE)
    reward_factor_list = [min(b / a, 10) for a, b in zip(percent_list, standard_percent_list)]
    return reward_factor_list


def main():
    seq_datasets = read_dataset_from_fasta()
    seq_datasets = process_data(seq_datasets)

    reward_factor_list = calc_reward_factor(seq_datasets)
    print(reward_factor_list)


if __name__ == "__main__":
    main()
