import numpy as np

from src.mean.ita_train import prepare_efficient_mc_att


def main():
    mode = "111"
    test_set = "../data/interim/mean/summaries/skempi_all.json"
    dataset, _ = prepare_efficient_mc_att(mode, test_set)
    origin_cplx = [dataset.data[i] for i in dataset.idx_mapping]

    list_sabdab_cdrh3pos = []
    for i in origin_cplx:
        list_sabdab_cdrh3pos.append(i.cdr_pos["CDR-H3"])
    print("list_sabdab_cdrh3pos: ", list_sabdab_cdrh3pos)

    np.save("../data/interim/list_sabdab_cdrh3pos.npy", list_sabdab_cdrh3pos)


if __name__ == '__main__':
    main()
