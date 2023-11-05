import time
import torch
import numpy as np
import tqdm

from src.utils_esmfold_plddt import esmfold_sequence_2_plddt, create_model
from src.utils_basic import wait_gpu_cool


def enable_cpu_offloading(model):
    from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel
    from torch.distributed.fsdp.wrap import enable_wrap, wrap

    torch.distributed.init_process_group(
        backend="nccl", init_method="tcp://localhost:9999", world_size=1, rank=0
    )

    wrapper_kwargs = dict(cpu_offload=CPUOffload(offload_params=True))

    with enable_wrap(wrapper_cls=FullyShardedDataParallel, **wrapper_kwargs):
        for layer_name, layer in model.layers.named_children():
            wrapped_layer = wrap(layer)
            setattr(model.layers, layer_name, wrapped_layer)
        model = wrap(model)

    return model


def init_model_on_gpu_with_cpu_offloading(model):
    model = model.eval()
    model_esm = enable_cpu_offloading(model.esm)
    del model.esm
    model.cuda()
    model.esm = model_esm
    return model


def main_single():
    # SEQUENCE = "MASMAKKDVIELEGTVSEALPNAMFKVKLENGHEILCHISGKLRMNFIRILEGDKVNVELSPYDLTRGRITWRKKLEHHHHHH"
    SEQUENCE = "AYVSDVYCEVCEFLVKEVTKLIDNNKTEKEILDAFDKMCSKLPKSLSEECQEVVDTYGSSILSILLEEVSPELVCSMLHLCSGTRHHHHHH:" \
               "AYVSDVYCEVCEFLVKEVTKLIDNNKTEKEILDAFDKMCSKLPKSLSEECQEVVDTYGSSILSILLEEVSPELVCSMLHLCSGTRHHHHHH"

    # SEQUENCE = "MLKNVLRYPGGKSKALKYILPNLPVGFREYREPMVGGGAVALAVKQLYTNVKIKINDLNYDLICFWKQLRDNPVQLIEEVSKIKENYKDGRKLYEFLTSQNGGGEFERAVRFYILNRITFSGTVDSGGYSQQSFENRFTWSAINKLKQAAEIIKDFEISHGDYEKLLFEPGNEVFIFLDPPYYSTTESRLYGKNGDLHLSFDHERFAFNIKKCPHLWMITYDDSPEVRKLFKFANIYEWELQYGMNNYKQSKAEKGKELFITNYKLEELRQKEKYALGLLEHHHHHH:" \
    #            "MLKNVLRYPGGKSKALKYILPNLPVGFREYREPMVGGGAVALAVKQLYTNVKIKINDLNYDLICFWKQLRDNPVQLIEEVSKIKENYKDGRKLYEFLTSQNGGGEFERAVRFYILNRITFSGTVDSGGYSQQSFENRFTWSAINKLKQAAEIIKDFEISHGDYEKLLFEPGNEVFIFLDPPYYSTTESRLYGKNGDLHLSFDHERFAFNIKKCPHLWMITYDDSPEVRKLFKFANIYEWELQYGMNNYKQSKAEKGKELFITNYKLEELRQKEKYALGLLEHHHHHH"

    # SEQUENCE = "EIQLQQSGAELVRPGALVKLSCKASGFNIKDYYMHWVKQRPEQGLEWIGLIDPENGNTIYDPKFQGKASITADTSSNTAYLQLSSLTSEDTAVYYCYYYYYYYYYYWGQGTTLTVSSA:" \
    #            "DIKMTQSPSSMYASLGERVTITCKASQDIRKYLNWYQQKPWKSPKTLIYYATSLADGVPSRFSGSGSGQDYSLTISSLESDDTATYYCLQHGESPYTFGGGTKLEINRA:" \
    #            "TNTVAAYNLTWKSTNFKTILEWEPKPVNQVYTVQISTKSGDWKSKCFYTTDTECDLTDEIVKDVKQTYLARVFSYPAGNEPLYENSPEFTPYLETNLGQPTIQSFEQVGTKVNVTVEDERTLVRRNNTFLSLRDVFGKDLIYTLYYWKSSSSGKKTAKTNTNEFLIDVDKGENYCFSVQAVIPSRTVNRKSTDSPVECMG"

    print("Initializing...")
    esm_model = create_model(chunk_size=128)  # 1:9619/5.42  16:9619/1.54  128:9619/1.49  1023:9721/1.50

    print("Processing...")
    sequence = SEQUENCE
    print("sequence: ", sequence)
    print("length: ", len(sequence))
    start_time = time.time()
    plddt_entire, plddt_specify, _, _ = esmfold_sequence_2_plddt(esm_model, sequence, specify_rate=0.3)
    print("infer time: ", time.time()-start_time)
    print("plddt_entire: ", plddt_entire)
    print("plddt_specify: ", plddt_specify)


def main_npy():
    print("Initializing...")
    origin_heavy_list = np.load("../data/raw/origin_heavy_list.npy")
    new_heavy_list = np.load("../data/raw/new_heavy_list_random_100.npy", allow_pickle=True)
    cdrh3_list = np.load("../data/raw/cdrh3_list.npy")
    esm_model = create_model(chunk_size=128)  # 1:9619/5.42  16:9619/1.54  128:9619/1.49  1023:9721/1.50
    delta_plddt_list = []
    delta_plddt_cdrh3_list = []
    new_plddt_cdrh3_list = []

    print("Processing...")
    for protein_i in tqdm.tqdm(range(len(origin_heavy_list))):
        print("")
        cdrh3_start = cdrh3_list[protein_i][0]
        cdrh3_end = cdrh3_list[protein_i][1]

        origin_sequence = origin_heavy_list[protein_i]
        origin_plddt_entire, _, _, origin_plddt_acid_list = esmfold_sequence_2_plddt(esm_model, origin_sequence)
        origin_plddt_cdrh3 = origin_plddt_acid_list[cdrh3_start: cdrh3_end]
        origin_plddt_cdrh3 = np.mean(origin_plddt_cdrh3)
        # print("origin_plddt_entire: ", origin_plddt_entire)
        print("origin_plddt_cdrh3: ", origin_plddt_cdrh3)
        wait_gpu_cool(60)

        new_sequence = new_heavy_list[protein_i]
        new_plddt_entire, _, _, new_plddt_acid_list = esmfold_sequence_2_plddt(esm_model, new_sequence)
        new_plddt_cdrh3 = new_plddt_acid_list[cdrh3_start: cdrh3_end]
        new_plddt_cdrh3 = np.mean(new_plddt_cdrh3)
        # print("new_plddt_entire: ", new_plddt_entire)
        print("new_plddt_cdrh3: ", new_plddt_cdrh3)
        wait_gpu_cool(60)

        new_plddt_cdrh3_list.append(new_plddt_cdrh3)
        delta_plddt = new_plddt_entire - origin_plddt_entire
        delta_plddt_cdrh3 = new_plddt_cdrh3 - origin_plddt_cdrh3
        delta_plddt_list.append(delta_plddt)
        delta_plddt_cdrh3_list.append(delta_plddt_cdrh3)

        print("------------------------------------------------")

    print("np.mean(delta_plddt_list): ", np.mean(delta_plddt_list))
    print("np.std(delta_plddt_list): ", np.std(delta_plddt_list))

    print("np.mean(delta_plddt_cdrh3_list): ", np.mean(delta_plddt_cdrh3_list))
    print("np.std(delta_plddt_cdrh3_list): ", np.std(delta_plddt_cdrh3_list))

    print("np.mean(new_plddt_cdrh3_list): ", np.mean(new_plddt_cdrh3_list))
    print("np.std(new_plddt_cdrh3_list): ", np.std(new_plddt_cdrh3_list))


if __name__ == "__main__":
    # main_single()
    main_npy()
