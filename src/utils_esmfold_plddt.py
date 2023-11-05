import dataclasses
import io
import string
import time
from typing import Optional, Sequence

import esm
import numpy as np
import torch
from Bio.PDB import PDBParser
from biotite.structure import io as bsio
from openfold.np import residue_constants


@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    # Chain indices for multi-chain predictions
    chain_index: Optional[np.ndarray] = None

    # Optional remark about the protein. Included as a comment in output PDB
    # files
    remark: Optional[str] = None

    # Templates used to generate this protein (prediction-only)
    parents: Optional[Sequence[str]] = None

    # Chain corresponding to each parent
    parents_chain_index: Optional[Sequence[int]] = None


def from_pdb_string(pdb_str: str, chain_id: Optional[str] = None, total_length=None) -> Protein:
    """Takes a PDB string and constructs a Protein object.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If None, then the pdb file must contain a single chain (which
        will be parsed). If chain_id is specified (e.g. A), then only that chain
        is parsed.

    Returns:
      A new `Protein` parsed from the pdb contents.
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f"Only single model PDBs are supported. Found {len(models)} models."
        )
    model = models[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain in model:
        if (chain_id is not None and chain.id != chain_id):
            continue
        for res in chain:
            if res.id[2] != " ":
                raise ValueError(
                    f"PDB contains an insertion code at chain {chain.id} and residue "
                    f"index {res.id[1]}. These are not supported."
                )
            res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
            restype_idx = residue_constants.restype_order.get(
                res_shortname, residue_constants.restype_num
            )
            pos = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.0
                res_b_factors[
                    residue_constants.atom_order[atom.name]
                ] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
            aatype.append(restype_idx)
            while len(atom_positions) < res.id[1] - 1:
                atom_positions.append(np.zeros([37, 3]))
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)
        if total_length is not None:
            while len(atom_positions) < total_length:
                atom_positions.append(np.zeros([37, 3]))

    parents = None
    parents_chain_index = None
    if ("PARENT" in pdb_str):
        parents = []
        parents_chain_index = []
        chain_id = 0
        for l in pdb_str.split("\n"):
            if ("PARENT" in l):
                if (not "N/A" in l):
                    parent_names = l.split()[1:]
                    parents.extend(parent_names)
                    parents_chain_index.extend([
                        chain_id for _ in parent_names
                    ])
                chain_id += 1

    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(string.ascii_uppercase)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        b_factors=np.array(b_factors),
        parents=parents,
        parents_chain_index=parents_chain_index,
    )


def esmfold_sequence_2_structure(esm_model, sequence, save_name=None):
    try:
        # inference
        with torch.no_grad():
            output = esm_model.infer_pdb(sequence)

        # save
        file_name = "../data/interim/esm_temp.pdb"
        with open(file_name, "w") as f:
            f.write(output)
        if save_name is not None:
            with open(save_name, "w") as f:
                f.write(output)

        # read for plddt
        struct = bsio.load_structure(file_name, extra_fields=["b_factor"])

        # read for lddt
        with open(file_name) as f:  # result_esm_6c00
            content = f.read()
        # print("Processing...")
        protein = from_pdb_string(content, total_length=len(sequence))

    except UnboundLocalError:
        print("UnboundLocalError!!!")
        struct = None
        protein = None
    return struct, protein


def esmfold_seqs_2_strus(esm_model, seq_list):
    try:
        # inference
        with torch.no_grad():
            # start_time = time.time()
            outputs = esm_model.infer_pdbs(seq_list)  # time!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # print("batch infer time: ", time.time()-start_time)

        struct_list = []
        protein_list = []
        for i_output in outputs:
            # save
            file_name = "../data/interim/esm_temp.pdb"
            with open(file_name, "w") as f:
                f.write(i_output)

            # read for plddt
            struct = bsio.load_structure(file_name, extra_fields=["b_factor"])
            struct_list.append(struct)

            # read for lddt
            with open(file_name) as f:  # result_esm_6c00
                content = f.read()
            # print("Processing...")
            protein = from_pdb_string(content, total_length=len(seq_list[0]))
            protein_list.append(protein)

    except UnboundLocalError:
        print("UnboundLocalError!!!")
        struct_list = None
        protein_list = None
    return struct_list, protein_list


def esmfold_structure_2_plddt(struct):
    if struct is not None:
        # plddt_atom
        plddt_atom = struct.b_factor
        # plddt_acid_array
        acid_num = struct.res_id[-1]
        plddt_acid_array = []
        for _ in range(acid_num):
            plddt_acid_array.append([])
        for atom_index in range(len(plddt_atom)):
            plddt_acid_array[struct.res_id[atom_index] - 1].append(plddt_atom[atom_index])
        # plddt_acid
        plddt_acid = []
        for i_acid in plddt_acid_array:
            if len(i_acid) > 0:
                temp_plddt_acid = np.mean(i_acid)
                assert type(temp_plddt_acid) is np.float64
                # assert temp_plddt_acid is not np.float("nan")
                plddt_acid.append(temp_plddt_acid)
            else:
                plddt_acid.append(1)  # temp_plddt_acid 1 0  # todo: decide which type
        # plddt
        plddt = np.mean(plddt_acid)
    else:
        print("maybe UnboundLocalError...")
        plddt = None
        plddt_acid = None
    return plddt, plddt_acid


def esmfold_strus_2_plddt(struct_list):
    plddt_list = []
    plddt_acid_list = []
    for i_struct in struct_list:
        if i_struct is not None:
            # plddt_atom
            plddt_atom = i_struct.b_factor
            # plddt_acid_array
            acid_num = i_struct.res_id[-1]
            plddt_acid_array = []
            for _ in range(acid_num):
                plddt_acid_array.append([])
            for atom_index in range(len(plddt_atom)):
                plddt_acid_array[i_struct.res_id[atom_index] - 1].append(plddt_atom[atom_index])
            # plddt_acid
            plddt_acid = []
            for i_acid in plddt_acid_array:
                if len(i_acid) > 0:
                    temp_plddt_acid = np.mean(i_acid)
                    assert type(temp_plddt_acid) is np.float64
                    # assert temp_plddt_acid is not np.float("nan")
                    plddt_acid.append(temp_plddt_acid)
                else:
                    plddt_acid.append(1)  # temp_plddt_acid 1 0  # todo: decide which type
            # plddt
            plddt = np.mean(plddt_acid)
        else:
            print(
                "maybe UnboundLocalError...",
            )
            plddt = None
            plddt_acid = None
        plddt_list.append(plddt)
        plddt_acid_list.append(plddt_acid)
    return plddt_list, plddt_acid_list


def esmfold_sequence_2_plddt(esm_model, sequence, specify_rate=None):
    struct, protein = esmfold_sequence_2_structure(esm_model, sequence)
    if struct is None:
        print("struct is None!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return 50, 0, 0, [0]
    plddt_entire, plddt_acid_list = esmfold_structure_2_plddt(struct)
    if specify_rate is not None:
        specify_acid_num = int(len(plddt_acid_list) * specify_rate)
        plddt_specify = np.mean(plddt_acid_list[:specify_acid_num])
        plddt_design = np.mean(plddt_acid_list[specify_acid_num:])
    else:
        plddt_specify = None
        plddt_design = None
    return plddt_entire, plddt_specify, plddt_design, plddt_acid_list


def esmfold_seqs_2_plddt(esm_model, seq_list):
    struct_list, protein_list = esmfold_seqs_2_strus(esm_model, seq_list)  # time!!!
    plddt_list, plddt_acid_list = esmfold_strus_2_plddt(struct_list)
    return plddt_list


def create_model(chunk_size=128):
    model = esm.pretrained.esmfold_v1()
    model = model.eval().cuda()
    model.set_chunk_size(chunk_size)
    return model
