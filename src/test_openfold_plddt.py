import time
import numpy as np
import utils_basic

from src.dir_openfold.utils_openfold import openfold_name_2_seqtag, openfold_seqtag_2_structure, calc_openfold_lddt, calc_openfold_plddt


def main_single():
    # 4NX7_A: MISLIAALAVDRVIGMENAMPWNLPADLAWFKRNTLNKPVIMGRHTWESIGRPLPGRKNIILSSQPGTDDRVTWVKSVDEAIAACGDVPEIMVIGGGRVYEQFLPKAQKLYLTHIDAEVEGDTHFPDYEPDDWESVFSEFHDADAQNSHSYCFEILERR
    # 1YSY_A: GHSKMSDVKCTSVVLLSVLQQLRVESSSKLWAQCVQLHNDILLAKDTTEAFEKMVSLLSVLLSMQGAVDINRLCEEMLDNRATLQ
    # >2QYP_A
    # AYVSDVYCEVCEFLVKEVTKLIDNNKTEKEILDAFDKMCSKLPKSLSEECQEVVDTYGSSILSILLEEVSPELVCSMLHLCSGTRHHHHHH
    # >2QYP_B
    # AYVSDVYCEVCEFLVKEVTKLIDNNKTEKEILDAFDKMCSKLPKSLSEECQEVVDTYGSSILSILLEEVSPELVCSMLHLCSGTRHHHHHH

    # seq_list = [["MISLIAALAVDRVIGMENAMPWNLPADLAWFKRNTLNKPVIMGRHTWESIGRPLPGRKNIILSSQPGTDDRVTWVKSVDEAIAACGDVPEIMVIGGGRVYEQFLPKAQKLYLTHIDAEVEGDTHFPDYEPDDWESVFSEFHDADAQNSHSYCFEILERR"]]
    # tag_list = [("4NX7", ["4NX7_A"])]

    # seq_list = [["AYVSDVYCEVCEFLVKEVTKLIDNNKTEKEILDAFDKMCSKLPKSLSEECQEVVDTYGSSILSILLEEVSPELVCSMLHLCSGTRHHHHHH",
    #              "AYVSDVYCEVCEFLVKEVTKLIDNNKTEKEILDAFDKMCSKLPKSLSEECQEVVDTYGSSILSILLEEVSPELVCSMLHLCSGTRHHHHHH"]]
    # tag_list = [("2QYP", ["2QYP_A", "2QYP_B"])]

    seq_list = [[# "DIKMTQSPSSMYASLGERVTITCKASQDIRKYLNWYQQKPWKSPKTLIYYATSLADGVPSRFSGSGSGQDYSLTISSLESDDTATYYCLQHGESPYTFGGGTKLEINRA",  # A D
                 "EIQLQQSGAELVRPGALVKLSCKASGFNIKDYYMHWVKQRPEQGLEWIGLIDPENGNTIYDPKFQGKASITADTSSNTAYLQLSSLTSEDTAVYYCARDNSYYFDYWGQGTTLTVSSA",  # B E
                 # "TNTVAAYNLTWKSTNFKTILEWEPKPVNQVYTVQISTKSGDWKSKCFYTTDTECDLTDEIVKDVKQTYLARVFSYPAGNEPLYENSPEFTPYLETNLGQPTIQSFEQVGTKVNVTVEDERTLVRRNNTFLSLRDVFGKDLIYTLYYWKSSSSGKKTAKTNTNEFLIDVDKGENYCFSVQAVIPSRTVNRKSTDSPVECMG",  # C F
                 # "ARDNSYYFDY",
                 ]]
    tag_list = [("1AHW", [# "1AHW_A",
                          "1AHW_B",
                          # "1AHW_C",
                          # "1AHW_CDRH3",
                          ])]
    print("Inferring...")
    start_time = time.time()
    output_protein_structure = openfold_seqtag_2_structure(seq_list, tag_list, save_pdb_flag=True, gpu_temp_limit=58)
    print("infer time: ", time.time()-start_time)

    print("Evaluating...")
    mean_plddt = calc_openfold_plddt(output_protein_structure)
    print("mean_plddt: ", mean_plddt)


def main_multi():
    new_heavy_seq_list = np.load("../data/raw/new_heavy_list_rl.npy", allow_pickle=True)
    tag_array = [("1AHW", ["1AHW_E"]),
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
    for i in range(len(tag_array)):
        seq_list = [[new_heavy_seq_list[i]]]
        tag_list = [tag_array[i]]
        output_protein_structure = openfold_seqtag_2_structure(seq_list, tag_list,
                                                               save_pdb_name="../data/interim/openfold/openfold_" + str(i) + "_" + tag_list[0][0] + ".pdb",
                                                               gpu_temp_limit=58)
        utils_basic.wait_gpu_cool(58)


if __name__ == "__main__":
    # main_single()
    main_multi()
