import random
import numpy as np
import tqdm

from src.mean.evaluation.pred_ddg import pred_ddg_only
from src.mean.generate import set_cdr
from src.mean.ita_train import prepare_efficient_mc_att

AC_TABLE = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'K', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


def seq_2_cplx(seq, origin_cplx_temp, id="1ahw"):
    # 'EIQLQQSGAELVRPGALVKLSCKASGFNIKDYYMHWVKQRPEQGLEWIGLIDPENGNTIYDPKFQGKASITADTSSNTAYLQLSSLTSEDTAVYYCYYYYYYYYYYWGQGTTLTVSSA'
    # 'DIKMTQSPSSMYASLGERVTITCKASQDIRKYLNWYQQKPWKSPKTLIYYATSLADGVPSRFSGSGSGQDYSLTISSLESDDTATYYCLQHGESPYTFGGGTKLEINRA'
    # 'TNTVAAYNLTWKSTNFKTILEWEPKPVNQVYTVQISTKSGDWKSKCFYTTDTECDLTDEIVKDVKQTYLARVFSYPAGNEPLYENSPEFTPYLETNLGQPTIQSFEQVGTKVNVTVEDERTLVRRNNTFLSLRDVFGKDLIYTLYYWKSSSSGKKTAKTNTNEFLIDVDKGENYCFSVQAVIPSRTVNRKSTDSPVECMG'

    x = np.load("../data/raw/true_x/true_x_" + id + ".npy")
    new_cplx = set_cdr(origin_cplx_temp, seq, x, cdr='H' + str(3))
    return new_cplx

def pdb_2_ddg(pdb_path,
              origin_cplx_path='/home/gemhou/Study/hj_mean_demo/summaries/ckpt/mean_CDR3_111/version_1/checkpoint/ita_results/ita_results/original/1ahw.pdb'):
    # origin_cplx_path
    # pdb_path = '/home/gemhou/Study/hj_mean_demo/summaries/ckpt/mean_CDR3_111/version_1/checkpoint/ita_results/ita_results/optimized/1ahw_2.pdb'
    score_ddp = pred_ddg_only(origin_cplx_path, pdb_path)
    return score_ddp


def seq_2_ddg(origin_cplx_temp, seq):
    id = origin_cplx_temp.get_id()
    new_cplx = seq_2_cplx(seq, origin_cplx_temp, id)
    pdb_path = "../data/raw/mean_temp.pdb"  # os.path.join(res_dir, new_cplx.get_id() + f'_{n}.pdb')
    new_cplx.to_pdb(pdb_path)
    score_ddp = pdb_2_ddg(pdb_path,
                          origin_cplx_path='/home/gemhou/Study/hj_mean_demo/summaries/ckpt/mean_CDR3_111/version_1/checkpoint/ita_results/ita_results/original/' + id + '.pdb')
    return score_ddp


def main_single():
    origin_index = 0  #   random.randint(0, 52)

    mode = "111"
    test_set = "../data/interim/mean/summaries/skempi_all.json"
    dataset, _ = prepare_efficient_mc_att(mode, test_set)
    origin_cplx = [dataset.data[i] for i in dataset.idx_mapping]

    origin_cplx_temp = origin_cplx[origin_index]

    id = origin_cplx_temp.get_id()
    x = np.load("../data/raw/true_x/true_x_" + id + ".npy")
    ddrh3_len = x.shape[0]
    seq = "CCCCCCCCCC"  # "A" * ddrh3_len  # TGYYYYYGDY YYYYYYYYYY TTTTTTTTT DEYWYYYYKW CCCCCCCCCC

    score_ddp = seq_2_ddg(origin_cplx_temp, seq)
    print("score_ddp: ", score_ddp)


def main_ac_table():
    """

    Print Results:
        seq:  AAAAAAAAAA  --- score_ddp:  5.8079023361206055
        seq:  CCCCCCCCCC  --- score_ddp:  2.136958360671997
        seq:  DDDDDDDDDD  --- score_ddp:  -1.4795721769332886
        seq:  EEEEEEEEEE  --- score_ddp:  1.2114932537078857
        seq:  FFFFFFFFFF  --- score_ddp:  -2.5540108680725098
        seq:  GGGGGGGGGG  --- score_ddp:  3.3335177898406982
        seq:  HHHHHHHHHH  --- score_ddp:  -0.23212486505508423
        seq:  IIIIIIIIII  --- score_ddp:  0.2696232199668884
        seq:  LLLLLLLLLL  --- score_ddp:  -0.7867012619972229
        seq:  KKKKKKKKKK  --- score_ddp:  1.0697447061538696
        seq:  MMMMMMMMMM  --- score_ddp:  0.35313355922698975
        seq:  NNNNNNNNNN  --- score_ddp:  1.9572296142578125
        seq:  PPPPPPPPPP  --- score_ddp:  3.4365932941436768
        seq:  QQQQQQQQQQ  --- score_ddp:  0.2240813672542572
        seq:  RRRRRRRRRR  --- score_ddp:  0.24466904997825623
        seq:  SSSSSSSSSS  --- score_ddp:  2.179490327835083
        seq:  TTTTTTTTTT  --- score_ddp:  -2.2440528869628906
        seq:  VVVVVVVVVV  --- score_ddp:  0.9381042718887329
        seq:  WWWWWWWWWW  --- score_ddp:  -3.6051204204559326
        seq:  YYYYYYYYYY  --- score_ddp:  -3.6710410118103027

    """
    for str_value in AC_TABLE:
        seq = str_value * 10
        mode = "111"
        test_set = "../data/interim/mean/summaries/skempi_all.json"
        dataset, _ = prepare_efficient_mc_att(mode, test_set)
        origin_cplx = [dataset.data[i] for i in dataset.idx_mapping]
        origin_cplx_temp = origin_cplx[0]
        new_cplx = seq_2_cplx(seq, origin_cplx_temp)
        pdb_path = "../data/raw/mean_temp.pdb"  # os.path.join(res_dir, new_cplx.get_id() + f'_{n}.pdb')
        new_cplx.to_pdb(pdb_path)
        score_ddp = pdb_2_ddg(pdb_path)
        print("seq: ", seq, " --- score_ddp: ", score_ddp)


def generate_seq_randomly(seq_length):
    seq = ""
    for ac_j in range(seq_length):
        j_index = random.randint(0, len(AC_TABLE)-1)
        j_str = AC_TABLE[j_index]
        seq = seq + j_str
    return seq


def main_random(seq_num=100):
    ddp_list = []
    for seq_i in tqdm.tqdm(range(seq_num)):
        seq = generate_seq_randomly(seq_length=10)
        mode = "111"
        test_set = "../data/interim/mean/summaries/skempi_all.json"
        dataset, _ = prepare_efficient_mc_att(mode, test_set)
        origin_cplx = [dataset.data[i] for i in dataset.idx_mapping]
        origin_cplx_temp = origin_cplx[0]
        new_cplx = seq_2_cplx(seq, origin_cplx_temp)
        pdb_path = "../data/raw/mean_temp.pdb"  # os.path.join(res_dir, new_cplx.get_id() + f'_{n}.pdb')
        new_cplx.to_pdb(pdb_path)
        score_ddp = pdb_2_ddg(pdb_path)
        print("")
        print("seq: ", seq, " --- score_ddp: ", score_ddp)
        ddp_list.append(score_ddp)
    best_ddp = min(ddp_list)
    print("best_ddp: ", best_ddp)
    mean_ddp = sum(ddp_list) / len(ddp_list)
    print("mean_ddp: ", mean_ddp)


def main():
    main_single()
    # main_ac_table()
    # main_random()


if __name__ == '__main__':
    main()
