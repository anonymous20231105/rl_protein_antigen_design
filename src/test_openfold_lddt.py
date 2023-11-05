import time

from src.dir_openfold.utils_openfold import openfold_name_2_seqtag, openfold_seqtag_2_structure, calc_openfold_lddt, calc_openfold_plddt

# 4NX7_A: MISLIAALAVDRVIGMENAMPWNLPADLAWFKRNTLNKPVIMGRHTWESIGRPLPGRKNIILSSQPGTDDRVTWVKSVDEAIAACGDVPEIMVIGGGRVYEQFLPKAQKLYLTHIDAEVEGDTHFPDYEPDDWESVFSEFHDADAQNSHSYCFEILERR
# 1YSY_A: GHSKMSDVKCTSVVLLSVLQQLRVESSSKLWAQCVQLHNDILLAKDTTEAFEKMVSLLSVLLSMQGAVDINRLCEEMLDNRATLQ
# >2QYP_A
# AYVSDVYCEVCEFLVKEVTKLIDNNKTEKEILDAFDKMCSKLPKSLSEECQEVVDTYGSSILSILLEEVSPELVCSMLHLCSGTRHHHHHH
# >2QYP_B
# AYVSDVYCEVCEFLVKEVTKLIDNNKTEKEILDAFDKMCSKLPKSLSEECQEVVDTYGSSILSILLEEVSPELVCSMLHLCSGTRHHHHHH

# seq_list = [["MISLIAALAVDRVIGMENAMPWNLPADLAWFKRNTLNKPVIMGRHTWESIGRPLPGRKNIILSSQPGTDDRVTWVKSVDEAIAACGDVPEIMVIGGGRVYEQFLPKAQKLYLTHIDAEVEGDTHFPDYEPDDWESVFSEFHDADAQNSHSYCFEILERR"]]
# tag_list = [("4NX7", ["4NX7_A"])]
# protein_name = "4nx7"  # 1prw 1ysy 4nx7 6c00 2qyp
# chain_id_list = ["A"]

# seq_list = [["AYVSDVYCEVCEFLVKEVTKLIDNNKTEKEILDAFDKMCSKLPKSLSEECQEVVDTYGSSILSILLEEVSPELVCSMLHLCSGTRHHHHHH",
#              "AYVSDVYCEVCEFLVKEVTKLIDNNKTEKEILDAFDKMCSKLPKSLSEECQEVVDTYGSSILSILLEEVSPELVCSMLHLCSGTRHHHHHH"]]
# tag_list = [("2QYP", ["2QYP_A", "2QYP_B"])]
# protein_name = "2qyp"  # 1prw 1ysy 4nx7 6c00 2qyp
# chain_id_list = ["A", "B"]

seq_list = [["DIVLTQSPASLSASVGETVTITCRASGNIHNYLAWYQQKQGKSPQLLVYYTTTLADGVPSRFSGSGSGTQYSLKINSLQPEDFGSYYCQHFWSTPRTFGGGTKLEIKR",
             # "QVQLQESGPGLVAPSQSLSITCTVSGFSLTGYGVNWVRQPPGKGLEWLGMIWGDGNTDYNSALKSRLSISKDNSKSQVFLKMNSLHTDDTARYYCARERDYRLDYWGQGTTLTVSS",
             # "DIQLTQSPSSLSASLGDRVTISCRASQDISNYLNWYQQKPDGTVKLLIYYTSRLHSGVPSRFSGSGSGTDYSLTISNLEQEDIATYFCQQGNTLPWTFGGGTKLEIK",
             # "QVQLQQSGTELVKSGASVKLSCTASGFNIKDTHMNWVKQRPEQGLEWIGRIDPANGNIQYDPKFRGKATITADTSSNTAYLQLSSLTSEDTAVYYCATKVIYYQGRGAMDYWGQGTTLTVS",
             ]]
tag_list = [("1DVF", ["1DVF_A",
                      # "1DVF_B",
                      # "1DVF_C",
                      # "1DVF_D",
                      ])]
protein_name = "1dvf"
chain_id_list = ["A",
                 # "B",
                 # "C",
                 # "D",
                 ]

MAX_RECYCLING_ITERS = 4
PLDDT_STOP = True
GPU_TEMP_LIMIT = 52


def main():
    print("Inferring...")

    start_time = time.time()
    output_protein_structure = \
        openfold_seqtag_2_structure(seq_list, tag_list, save_pdb_name="../data/interim/temp.pdb", max_recycling_iters=MAX_RECYCLING_ITERS,
                                    gpu_temp_limit=GPU_TEMP_LIMIT, plddt_stop=PLDDT_STOP)
    print("infer time: ", time.time()-start_time)

    print("Evaluating...")
    mean_plddt = calc_openfold_plddt(output_protein_structure)
    print("mean_plddt: ", mean_plddt)

    mean_lddt = calc_openfold_lddt(output_protein_structure, protein_name, chain_id_list=chain_id_list)
    print("mean_lddt: ", mean_lddt)


if __name__ == "__main__":
    main()
