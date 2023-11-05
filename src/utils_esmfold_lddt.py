from typing import List

import numpy as np
import torch
from openfold.data import mmcif_parsing
from openfold.np import residue_constants

from src.utils_esmfold_plddt import esmfold_sequence_2_structure


def generate_all_atom_positions(protein_name):
    chain_id = "A"
    file_id = protein_name
    path = "../data/interim/mmcif/mmcif_" + protein_name + "/" + protein_name + ".cif"
    with open(path, "r") as f:
        mmcif_string = f.read()
    mmcif_object = mmcif_parsing.parse(file_id=file_id, mmcif_string=mmcif_string)
    mmcif_object = mmcif_object.mmcif_object
    all_atom_positions, all_atom_mask = mmcif_parsing.get_atom_coords(
        mmcif_object=mmcif_object, chain_id=chain_id
    )
    gt_coords = all_atom_positions
    gt_coords = torch.tensor(gt_coords)
    all_atom_mask = all_atom_mask
    all_atom_mask = torch.tensor(all_atom_mask)
    all_atom_positions = gt_coords
    return all_atom_mask, all_atom_positions


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def lddt(
        all_atom_pred_pos: torch.Tensor,
        all_atom_positions: torch.Tensor,
        all_atom_mask: torch.Tensor,
        cutoff: float = 15.0,
        eps: float = 1e-10,
        per_residue: bool = True,
) -> torch.Tensor:
    all_atom_mask_numpy = all_atom_mask.numpy()
    all_atom_mask_numpy = all_atom_mask_numpy.squeeze()

    n = all_atom_mask.shape[-2]
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (
                    all_atom_positions[..., None, :]
                    - all_atom_positions[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (
                    all_atom_pred_pos[..., None, :]
                    - all_atom_pred_pos[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    dmat_true_numpy = dmat_true.numpy()
    dmat_pred_numpy = dmat_pred.numpy()

    dists_to_score = (
            (dmat_true < cutoff)
            * all_atom_mask
            * permute_final_dims(all_atom_mask, (1, 0))
            * (1.0 - torch.eye(n, device=all_atom_mask.device))
    )
    dists_to_score_numpy = dists_to_score.numpy()

    # dmat_pred.to("cuda:0")
    # dmat_true.to("cuda:0")

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    dist_l1_numpy = dist_l1.numpy()

    score1 = (
            (dist_l1 < 0.5).type(dist_l1.dtype)
            + (dist_l1 < 1.0).type(dist_l1.dtype)
            + (dist_l1 < 2.0).type(dist_l1.dtype)
            + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score1_numpy = score1.numpy()

    score2 = score1 * 0.25
    score2_numpy = score2.numpy()

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score3 = norm * (eps + torch.sum(dists_to_score * score2, dim=dims))
    score3_numpy = score3.numpy()

    return score3 * 100


def lddt_ca(
        all_atom_pred_pos: torch.Tensor,
        all_atom_positions: torch.Tensor,
        all_atom_mask: torch.Tensor,
        cutoff: float = 15.0,
        eps: float = 1e-10,
        per_residue: bool = True,
) -> torch.Tensor:
    ca_pos = residue_constants.atom_order["CA"]
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    all_atom_positions = all_atom_positions[..., ca_pos, :]
    all_atom_mask = all_atom_mask[..., ca_pos: (ca_pos + 1)]  # keep dim

    return lddt(
        all_atom_pred_pos,
        all_atom_positions,
        all_atom_mask,
        cutoff=cutoff,
        eps=eps,
        per_residue=per_residue,
    )


def calc_lddt_ca(pred_coords, all_atom_positions, all_atom_mask, mut_rate):
    pred_coords = pred_coords.to("cpu")
    lddt_ca_score = lddt_ca(
        pred_coords,
        all_atom_positions,
        all_atom_mask,
    )
    lddt_ca_list = lddt_ca_score.numpy()

    length = len(lddt_ca_list)
    lddt_ca_list_origin = lddt_ca_list[: int(length * (1 - mut_rate))]
    lddt_ca_list_pred = lddt_ca_list[int(length * (1 - mut_rate)):]

    lddt_ca_mean = np.mean(lddt_ca_list)
    lddt_ca_origin = np.mean(lddt_ca_list_origin)
    lddt_ca_pred = np.mean(lddt_ca_list_pred)
    return lddt_ca_mean, lddt_ca_origin, lddt_ca_pred


def pos_name_2_lddt(pred_coords, protein_name, mut_rate=0):
    all_atom_mask, all_atom_positions = generate_all_atom_positions(
        protein_name
    )
    cut_rate = 0
    total_len = int(all_atom_positions.shape[0])
    given_len = int(total_len * (1 - cut_rate))
    pred_len = total_len - given_len
    all_atom_positions_given, all_atom_positions_pred = torch.split(
        all_atom_positions, [given_len, pred_len]
    )
    all_atom_mask_given, all_atom_mask_pred = torch.split(
        all_atom_mask, [given_len, pred_len]
    )
    lddt_ca_score_mean, lddt_ca_origin, lddt_ca_pred = calc_lddt_ca(
        pred_coords, all_atom_positions_given, all_atom_mask_given, mut_rate
    )
    return lddt_ca_origin, lddt_ca_pred, lddt_ca_score_mean


def esmfold_sequence_2_lddt(esm_model, protein_name, sequence):
    struct, protein = esmfold_sequence_2_structure(esm_model, sequence)
    pred_coords = torch.tensor(protein.atom_positions)
    lddt_ca_origin, lddt_ca_pred, lddt_ca_score_mean = pos_name_2_lddt(pred_coords, protein_name)
    return lddt_ca_score_mean
