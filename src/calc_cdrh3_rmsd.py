import copy

import numpy as np
import tqdm

from src.dir_openfold.utils_openfold import openfold_seqtag_2_structure
import utils_basic
import src.utils_rmsd


def calc_sabdab_heavy_atom37pos(list_sabdab_heavy_seq):
    sabdab_heavy_tag_array = [("1AHW", ["1AHW_E"]),
                              ("1DVF", ["1DVF_B"]),
                              ("1DFB", ["1DFB_B"]),
                              ("2VIS", ["2VIS_B"]),
                              ("2VIR", ["2VIR_B"]),
                              ("1KIQ", ["1KIQ_B"]),
                              ("1KIP", ["1KIP_B"]),
                              ("1KIR", ["1KIR_B"]),
                              ("2JEL", ["2JEL_H"]),
                              ("1NCA", ["1NCA_H"]),
                              ("1DQJ", ["1DQJ_B"]),
                              ("1JRH", ["1JRH_H"]),
                              ("1NMB", ["1NMB_H"]),
                              ("3HFM", ["3HFM_H"]),
                              ("1YY9", ["1YY9_D"]),
                              ("4GXU", ["4GXU_M"]),
                              ("3LZF", ["3LZF_H"]),
                              ("1N8Z", ["1N8Z_B"]),
                              ("3G6D", ["3G6D_H"]),
                              ("1XGU", ["1XGU_B"]),
                              ("1XGP", ["1XGP_B"]),
                              ("1XGQ", ["1XGQ_B"]),
                              ("1XGR", ["1XGR_B"]),
                              ("1XGT", ["1XGT_B"]),
                              ("3N85", ["3N85_H"]),
                              ("4I77", ["4I77_H"]),
                              ("3L5X", ["3L5X_H"]),
                              ("4JPK", ["4JPK_H"]),
                              ("1BJ1", ["1BJ1_H"]),
                              ("1CZ8", ["1CZ8_Y"]),
                              ("1MHP", ["1MHP_X"]),
                              ("2B2X", ["2B2X_H"]),
                              ("1MLC", ["1MLC_B"]),
                              ("3BDY", ["3BDY_H"]),
                              ("3BE1", ["3BE1_H"]),
                              ("2NY7", ["2NY7_H"]),
                              ("3IDX", ["3IDX_H"]),
                              ("2NYY", ["2NYY_D"]),
                              ("2NZ9", ["2NZ9_F"]),
                              ("3NGB", ["3NGB_B"]),
                              ("2BDN", ["2BDN_H"]),
                              ("3W2D", ["3W2D_H"]),
                              ("4KRL", ["4KRL_B"]),
                              ("4KRO", ["4KRO_B"]),
                              ("4KRP", ["4KRP_B"]),
                              ("4NM8", ["4NM8_H"]),
                              ("4U6H", ["4U6H_A"]),
                              ("4ZS6", ["4ZS6_C"]),
                              ("5C6T", ["5C6T_H"]),
                              ("5DWU", ["5DWU_H"]),
                              ("3SE8", ["3SE8_H"]),
                              ("3SE9", ["3SE9_H"]),
                              ("1YQV", ["1YQV_H"]),
                              ]
    list_sabdab_heavy_atom37pos = []
    for i in tqdm.tqdm(range(len(sabdab_heavy_tag_array))):  # len(sabdab_heavy_tag_array) 2
        temp_seq_list = [[list_sabdab_heavy_seq[i]]]
        temp_tag_list = [sabdab_heavy_tag_array[i]]
        output_protein_structure = openfold_seqtag_2_structure(temp_seq_list, temp_tag_list,
                                                               save_pdb_name=None,
                                                               gpu_temp_limit=58)  # time!!!!!!!!!!!!!!!!!!!!!!!!!!
        tensor_sabdab_heavy_atom37pos = output_protein_structure["final_atom_positions"]
        array_sabdab_heavy_atom37pos = tensor_sabdab_heavy_atom37pos.to("cpu").numpy()
        list_sabdab_heavy_atom37pos.append(array_sabdab_heavy_atom37pos)
        utils_basic.wait_gpu_cool(58)
    return list_sabdab_heavy_atom37pos


def calc_sabdab_heavy_atom4pos(list_sabdab_heavy_atom37pos):
    list_sabdab_heavy_atom4pos = copy.deepcopy(list_sabdab_heavy_atom37pos)
    for i in range(len(list_sabdab_heavy_atom4pos)):
        list_sabdab_heavy_atom4pos[i] = list_sabdab_heavy_atom4pos[i][:, [0, 1, 2, 4], :]
    return list_sabdab_heavy_atom4pos


def calc_sabdab_cdrh3_atom4pos(list_sabdab_cdrh3pos, list_sabdab_heavy_atom4pos):
    list_sabdab_cdrh3_atom4pos = copy.deepcopy(list_sabdab_heavy_atom4pos)
    for i in range(len(list_sabdab_cdrh3_atom4pos)):
        list_sabdab_cdrh3_atom4pos[i] = list_sabdab_cdrh3_atom4pos[i][
                                            list_sabdab_cdrh3pos[i][0]: list_sabdab_cdrh3pos[i][1] + 1, :, :]
    return list_sabdab_cdrh3_atom4pos


def calc_sabdab_cdrh3_rmsd(list_sabdab_cdrh3_atom4pos, list_sabdab_cdrh3_atom4pos_true):
    list_sabdab_cdrh3_rmsd = []
    for temp_method, temp_true in zip(list_sabdab_cdrh3_atom4pos, list_sabdab_cdrh3_atom4pos_true):
        temp_method_3 = temp_method.reshape(-1, 3)
        temp_true_3 = temp_true.reshape(-1, 3)
        temp_rmsd = src.utils_rmsd.compute_rmsd(temp_method_3, temp_true_3)
        print("temp_rmsd: ", temp_rmsd)
        list_sabdab_cdrh3_rmsd.append(temp_rmsd)
    return list_sabdab_cdrh3_rmsd


def main():
    method_name = "mean1"  # our mean100 mean1 random1 random100
    try:
        list_sabdab_heavy_atom37pos = np.load("../data/interim/list_sabdab_heavy_atom37pos_" + method_name + ".npy",
                                              allow_pickle=True)
    except FileNotFoundError:
        list_sabdab_heavy_seq = np.load("../data/interim/list_sabdab_heavy_seq_" + method_name + ".npy",
                                        allow_pickle=True)
        list_sabdab_heavy_atom37pos = calc_sabdab_heavy_atom37pos(list_sabdab_heavy_seq)  # time: 20 minutes
        np.save("../data/interim/list_sabdab_heavy_atom37pos_" + method_name + ".npy", list_sabdab_heavy_atom37pos)
    list_sabdab_heavy_atom4pos = calc_sabdab_heavy_atom4pos(list_sabdab_heavy_atom37pos)
    list_sabdab_cdrh3pos = np.load("../data/interim/list_sabdab_cdrh3pos.npy", allow_pickle=True)
    list_sabdab_cdrh3_atom4pos = calc_sabdab_cdrh3_atom4pos(list_sabdab_cdrh3pos, list_sabdab_heavy_atom4pos)
    list_sabdab_cdrh3_atom4pos_true = np.load("../data/interim/list_sabdab_cdrh3_atom4pos_true.npy", allow_pickle=True)
    list_sabdab_cdrh3_rmsd = calc_sabdab_cdrh3_rmsd(list_sabdab_cdrh3_atom4pos, list_sabdab_cdrh3_atom4pos_true)
    print("np.mean(list_sabdab_cdrh3_rmsd_" + method_name + "): ", np.mean(list_sabdab_cdrh3_rmsd))
    print("np.std(list_sabdab_cdrh3_rmsd_" + method_name + "): ", np.std(list_sabdab_cdrh3_rmsd))
    print("Finished...")


if __name__ == "__main__":
    main()
