import time
import torch
import numpy as np
import tqdm

from src.utils_esmfold_plddt import esmfold_sequence_2_plddt, create_model
from src.utils_basic import wait_gpu_cool
from src.mean.ita_train import prepare_efficient_mc_att


def main_npy():
    print("Initializing...")
    origin_heavy_list = np.load("../data/raw/origin_heavy_list.npy")
    new_heavy_list = np.load("../data/raw/new_heavy_list_mean.npy", allow_pickle=True)
    cdrh3_list = np.load("../data/raw/cdrh3_list.npy")
    # plddt
    esm_model = create_model(chunk_size=128)  # 1:9619/5.42  16:9619/1.54  128:9619/1.49  1023:9721/1.50
    # ddg
    mode = "111"
    test_set = "../data/interim/mean/summaries/skempi_all.json"
    dataset, _ = prepare_efficient_mc_att(mode, test_set)
    origin_cplx = [dataset.data[i] for i in dataset.idx_mapping]

    print("Processing...")
    delta_plddt_list = []
    delta_plddt_cdrh3_list = []
    new_plddt_cdrh3_list = []
    ddg_list = []
    result_list = []
    for protein_i in tqdm.tqdm(range(len(origin_heavy_list))):
        print("")
        cdrh3_start = cdrh3_list[protein_i][0]
        cdrh3_end = cdrh3_list[protein_i][1]

        origin_sequence = origin_heavy_list[protein_i]
        origin_plddt_entire, _, _, origin_plddt_acid_list = esmfold_sequence_2_plddt(esm_model, origin_sequence)
        origin_plddt_cdrh3 = origin_plddt_acid_list[cdrh3_start: cdrh3_end]
        origin_plddt_cdrh3 = np.mean(origin_plddt_cdrh3)
        print("origin_plddt_cdrh3: ", origin_plddt_cdrh3)
        wait_gpu_cool(60)

        new_sequence = new_heavy_list[protein_i]
        new_plddt_entire, _, _, new_plddt_acid_list = esmfold_sequence_2_plddt(esm_model, new_sequence)

        origin_cplx_temp = origin_cplx[protein_i]
        from test_mean_ddg import seq_2_ddg
        seq = new_sequence[cdrh3_start: cdrh3_end]
        ddg = seq_2_ddg(origin_cplx_temp, seq)
        print("ddg: ", ddg)
        ddg_list.append(ddg)

        new_plddt_cdrh3 = new_plddt_acid_list[cdrh3_start: cdrh3_end]
        new_plddt_cdrh3 = np.mean(new_plddt_cdrh3)
        print("new_plddt_cdrh3: ", new_plddt_cdrh3)
        wait_gpu_cool(60)

        new_plddt_cdrh3_list.append(new_plddt_cdrh3)
        delta_plddt = new_plddt_entire - origin_plddt_entire
        delta_plddt_cdrh3 = new_plddt_cdrh3 - origin_plddt_cdrh3
        delta_plddt_list.append(delta_plddt)
        delta_plddt_cdrh3_list.append(delta_plddt_cdrh3)

        ddrh3_len = cdrh3_end - cdrh3_start

        result = [ddrh3_len, new_plddt_cdrh3, ddg]
        result_list.append(result)

        print("------------------------------------------------")

    print("np.mean(delta_plddt_list): ", np.mean(delta_plddt_list))
    print("np.std(delta_plddt_list): ", np.std(delta_plddt_list))

    print("np.mean(delta_plddt_cdrh3_list): ", np.mean(delta_plddt_cdrh3_list))
    print("np.std(delta_plddt_cdrh3_list): ", np.std(delta_plddt_cdrh3_list))

    print("np.mean(new_plddt_cdrh3_list): ", np.mean(new_plddt_cdrh3_list))
    print("np.std(new_plddt_cdrh3_list): ", np.std(new_plddt_cdrh3_list))

    print("np.mean(ddg_list): ", np.mean(ddg_list))
    print("np.std(ddg_list): ", np.std(ddg_list))

    np.save("../data/processed/result_mean_1.npy", result_list)


if __name__ == "__main__":
    main_npy()
