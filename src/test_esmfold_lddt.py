from src.utils_esmfold_lddt import esmfold_sequence_2_lddt
from src.utils_esmfold_plddt import create_model
import torch

SEQUENCE = "MASMAKKDVIELEGTVSEALPNAMFKVKLENGHEILCHISGKLRMNFIRILEGDKVNVELSPYDLTRGRITWRKKLEHHHHHH"
PROTEIN_NAME = "6c00"  # "6c00"


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


def main():
    print("Initializing...")
    esm_model = create_model()
    esm_model = init_model_on_gpu_with_cpu_offloading(esm_model)

    print("Processing...")
    sequence = SEQUENCE
    protein_name = PROTEIN_NAME
    print("sequence: ", sequence)
    print("length: ", len(sequence))

    lddt_ca_score_mean = esmfold_sequence_2_lddt(esm_model, protein_name, sequence)
    print("lddt: ", lddt_ca_score_mean)


if __name__ == "__main__":
    main()
