# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from multiprocessing import cpu_count
from typing import Mapping, Optional, Sequence, Any

import numpy as np

from openfold.data import templates, parsers, mmcif_parsing
from openfold.data.templates import get_custom_template_features
from openfold.data.tools import jackhmmer, hhblits, hhsearch
from openfold.data.tools.utils import to_date
from openfold.np import residue_constants
# from srcs import utils_openfold_protein as protein
import dataclasses
import io
from Bio.PDB import PDBParser
import string
import re

FeatureDict = Mapping[str, np.ndarray]


def empty_template_feats(n_res) -> FeatureDict:
    return {
        "template_aatype": np.zeros((0, n_res)).astype(np.int64),
        "template_all_atom_positions":
            np.zeros((0, n_res, 37, 3)).astype(np.float32),
        "template_sum_probs": np.zeros((0, 1)).astype(np.float32),
        "template_all_atom_mask": np.zeros((0, n_res, 37)).astype(np.float32),
    }


def make_template_features(
        input_sequence: str,
        hits: Sequence[Any],
        template_featurizer: Any,
        query_pdb_code: Optional[str] = None,
        query_release_date: Optional[str] = None,
) -> FeatureDict:
    hits_cat = sum(hits.values(), [])
    if (len(hits_cat) == 0 or template_featurizer is None):
        template_features = empty_template_feats(len(input_sequence))
    else:
        templates_result = template_featurizer.get_templates(
            query_sequence=input_sequence,
            query_pdb_code=query_pdb_code,
            query_release_date=query_release_date,
            hits=hits_cat,
        )
        template_features = templates_result.features

        # The template featurizer doesn't format empty template features
        # properly. This is a quick fix.
        if (template_features["template_aatype"].shape[0] == 0):
            template_features = empty_template_feats(len(input_sequence))

    return template_features


def unify_template_features(
        template_feature_list: Sequence[FeatureDict]
) -> FeatureDict:
    out_dicts = []
    seq_lens = [fd["template_aatype"].shape[1] for fd in template_feature_list]
    for i, fd in enumerate(template_feature_list):
        out_dict = {}
        n_templates, n_res = fd["template_aatype"].shape[:2]
        for k, v in fd.items():
            seq_keys = [
                "template_aatype",
                "template_all_atom_positions",
                "template_all_atom_mask",
            ]
            if (k in seq_keys):
                new_shape = list(v.shape)
                assert (new_shape[1] == n_res)
                new_shape[1] = sum(seq_lens)
                new_array = np.zeros(new_shape, dtype=v.dtype)

                if (k == "template_aatype"):
                    new_array[..., residue_constants.HHBLITS_AA_TO_ID['-']] = 1

                offset = sum(seq_lens[:i])
                new_array[:, offset:offset + seq_lens[i]] = v
                out_dict[k] = new_array
            else:
                out_dict[k] = v

        chain_indices = np.array(n_templates * [i])
        out_dict["template_chain_index"] = chain_indices

        if (n_templates != 0):
            out_dicts.append(out_dict)

    if (len(out_dicts) > 0):
        out_dict = {
            k: np.concatenate([od[k] for od in out_dicts]) for k in out_dicts[0]
        }
    else:
        out_dict = empty_template_feats(sum(seq_lens))

    return out_dict


def make_sequence_features(
        sequence: str, description: str, num_res: int
) -> FeatureDict:
    """Construct a feature dict of sequence features."""
    features = {}
    features["aatype"] = residue_constants.sequence_to_onehot(
        sequence=sequence,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True,
    )
    features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
    features["domain_name"] = np.array(
        [description.encode("utf-8")], dtype=np.object_
    )
    features["residue_index"] = np.array(range(num_res), dtype=np.int32)
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    features["sequence"] = np.array(
        [sequence.encode("utf-8")], dtype=np.object_
    )
    return features


def make_mmcif_features(
        mmcif_object: mmcif_parsing.MmcifObject, chain_id: str
) -> FeatureDict:
    input_sequence = mmcif_object.chain_to_seqres[chain_id]
    description = "_".join([mmcif_object.file_id, chain_id])
    num_res = len(input_sequence)

    mmcif_feats = {}

    mmcif_feats.update(
        make_sequence_features(
            sequence=input_sequence,
            description=description,
            num_res=num_res,
        )
    )

    all_atom_positions, all_atom_mask = mmcif_parsing.get_atom_coords(
        mmcif_object=mmcif_object, chain_id=chain_id
    )
    mmcif_feats["all_atom_positions"] = all_atom_positions
    mmcif_feats["all_atom_mask"] = all_atom_mask

    mmcif_feats["resolution"] = np.array(
        [mmcif_object.header["resolution"]], dtype=np.float32
    )

    mmcif_feats["release_date"] = np.array(
        [mmcif_object.header["release_date"].encode("utf-8")], dtype=np.object_
    )

    mmcif_feats["is_distillation"] = np.array(0., dtype=np.float32)

    return mmcif_feats


def _aatype_to_str_sequence(aatype):
    return ''.join([
        residue_constants.restypes_with_x[aatype[i]]
        for i in range(len(aatype))
    ])


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


def make_protein_features(
        protein_object: Protein,
        description: str,
        _is_distillation: bool = False,
) -> FeatureDict:
    pdb_feats = {}
    aatype = protein_object.aatype
    sequence = _aatype_to_str_sequence(aatype)
    pdb_feats.update(
        make_sequence_features(
            sequence=sequence,
            description=description,
            num_res=len(protein_object.aatype),
        )
    )

    all_atom_positions = protein_object.atom_positions
    all_atom_mask = protein_object.atom_mask

    pdb_feats["all_atom_positions"] = all_atom_positions.astype(np.float32)
    pdb_feats["all_atom_mask"] = all_atom_mask.astype(np.float32)

    pdb_feats["resolution"] = np.array([0.]).astype(np.float32)
    pdb_feats["is_distillation"] = np.array(
        1. if _is_distillation else 0.
    ).astype(np.float32)

    return pdb_feats


def make_pdb_features(
        protein_object: Protein,
        description: str,
        is_distillation: bool = True,
        confidence_threshold: float = 50.,
) -> FeatureDict:
    pdb_feats = make_protein_features(
        protein_object, description, _is_distillation=True
    )

    if (is_distillation):
        high_confidence = protein_object.b_factors > confidence_threshold
        high_confidence = np.any(high_confidence, axis=-1)
        pdb_feats["all_atom_mask"] *= high_confidence[..., None]

    return pdb_feats


def make_msa_features(
        msas: Sequence[Sequence[str]],
        deletion_matrices: Sequence[parsers.DeletionMatrix],
) -> FeatureDict:
    """Constructs a feature dict of MSA features."""
    if not msas:
        raise ValueError("At least one MSA must be provided.")

    int_msa = []
    deletion_matrix = []
    seen_sequences = set()
    for msa_index, msa in enumerate(msas):
        if not msa:
            raise ValueError(
                f"MSA {msa_index} must contain at least one sequence."
            )
        for sequence_index, sequence in enumerate(msa):
            if sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            int_msa.append(
                [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence]
            )
            deletion_matrix.append(deletion_matrices[msa_index][sequence_index])

    num_res = len(msas[0][0])
    num_alignments = len(int_msa)
    features = {}
    features["deletion_matrix_int"] = np.array(deletion_matrix, dtype=np.int32)
    features["msa"] = np.array(int_msa, dtype=np.int32)
    features["num_alignments"] = np.array(
        [num_alignments] * num_res, dtype=np.int32
    )
    return features


def make_sequence_features_with_custom_template(
        sequence: str,
        mmcif_path: str,
        pdb_id: str,
        chain_id: str,
        kalign_binary_path: str) -> FeatureDict:
    """
    process a single fasta file using features derived from a single template rather than an alignment
    """
    num_res = len(sequence)

    sequence_features = make_sequence_features(
        sequence=sequence,
        description=pdb_id,
        num_res=num_res,
    )

    msa_data = [[sequence]]
    deletion_matrix = [[[0 for _ in sequence]]]

    msa_features = make_msa_features(msa_data, deletion_matrix)
    template_features = get_custom_template_features(
        mmcif_path=mmcif_path,
        query_sequence=sequence,
        pdb_id=pdb_id,
        chain_id=chain_id,
        kalign_binary_path=kalign_binary_path
    )

    return {
        **sequence_features,
        **msa_features,
        **template_features.features
    }


class AlignmentRunner:
    """Runs alignment tools and saves the results"""

    def __init__(
            self,
            jackhmmer_binary_path: Optional[str] = None,
            hhblits_binary_path: Optional[str] = None,
            hhsearch_binary_path: Optional[str] = None,
            uniref90_database_path: Optional[str] = None,
            mgnify_database_path: Optional[str] = None,
            bfd_database_path: Optional[str] = None,
            uniclust30_database_path: Optional[str] = None,
            pdb70_database_path: Optional[str] = None,
            use_small_bfd: Optional[bool] = None,
            no_cpus: Optional[int] = None,
            uniref_max_hits: int = 10000,
            mgnify_max_hits: int = 5000,
    ):
        """
        Args:
            jackhmmer_binary_path:
                Path to jackhmmer binary
            hhblits_binary_path:
                Path to hhblits binary
            hhsearch_binary_path:
                Path to hhsearch binary
            uniref90_database_path:
                Path to uniref90 database. If provided, jackhmmer_binary_path
                must also be provided
            mgnify_database_path:
                Path to mgnify database. If provided, jackhmmer_binary_path
                must also be provided
            bfd_database_path:
                Path to BFD database. Depending on the value of use_small_bfd,
                one of hhblits_binary_path or jackhmmer_binary_path must be 
                provided.
            uniclust30_database_path:
                Path to uniclust30. Searched alongside BFD if use_small_bfd is 
                false.
            pdb70_database_path:
                Path to pdb70 database.
            use_small_bfd:
                Whether to search the BFD database alone with jackhmmer or 
                in conjunction with uniclust30 with hhblits.
            no_cpus:
                The number of CPUs available for alignment. By default, all
                CPUs are used.
            uniref_max_hits:
                Max number of uniref hits
            mgnify_max_hits:
                Max number of mgnify hits
        """
        db_map = {
            "jackhmmer": {
                "binary": jackhmmer_binary_path,
                "dbs": [
                    uniref90_database_path,
                    mgnify_database_path,
                    bfd_database_path if use_small_bfd else None,
                ],
            },
            "hhblits": {
                "binary": hhblits_binary_path,
                "dbs": [
                    bfd_database_path if not use_small_bfd else None,
                ],
            },
            "hhsearch": {
                "binary": hhsearch_binary_path,
                "dbs": [
                    pdb70_database_path,
                ],
            },
        }

        for name, dic in db_map.items():
            binary, dbs = dic["binary"], dic["dbs"]
            if (binary is None and not all([x is None for x in dbs])):
                raise ValueError(
                    f"{name} DBs provided but {name} binary is None"
                )

        if (not all([x is None for x in db_map["hhsearch"]["dbs"]])
                and uniref90_database_path is None):
            raise ValueError(
                """uniref90_database_path must be specified in order to perform
                   template search"""
            )

        self.uniref_max_hits = uniref_max_hits
        self.mgnify_max_hits = mgnify_max_hits
        self.use_small_bfd = use_small_bfd

        if (no_cpus is None):
            no_cpus = cpu_count()

        self.jackhmmer_uniref90_runner = None
        if (jackhmmer_binary_path is not None and
                uniref90_database_path is not None
        ):
            self.jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
                binary_path=jackhmmer_binary_path,
                database_path=uniref90_database_path,
                n_cpu=no_cpus,
            )

        self.jackhmmer_small_bfd_runner = None
        self.hhblits_bfd_uniclust_runner = None
        if (bfd_database_path is not None):
            if use_small_bfd:
                self.jackhmmer_small_bfd_runner = jackhmmer.Jackhmmer(
                    binary_path=jackhmmer_binary_path,
                    database_path=bfd_database_path,
                    n_cpu=no_cpus,
                )
            else:
                dbs = [bfd_database_path]
                if (uniclust30_database_path is not None):
                    dbs.append(uniclust30_database_path)
                self.hhblits_bfd_uniclust_runner = hhblits.HHBlits(
                    binary_path=hhblits_binary_path,
                    databases=dbs,
                    n_cpu=no_cpus,
                )

        self.jackhmmer_mgnify_runner = None
        if (mgnify_database_path is not None):
            self.jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
                binary_path=jackhmmer_binary_path,
                database_path=mgnify_database_path,
                n_cpu=no_cpus,
            )

        self.hhsearch_pdb70_runner = None
        if (pdb70_database_path is not None):
            self.hhsearch_pdb70_runner = hhsearch.HHSearch(
                binary_path=hhsearch_binary_path,
                databases=[pdb70_database_path],
                n_cpu=no_cpus,
            )

    def run(
            self,
            fasta_path: str,
            output_dir: str,
    ):
        """Runs alignment tools on a sequence"""
        if (self.jackhmmer_uniref90_runner is not None):
            jackhmmer_uniref90_result = self.jackhmmer_uniref90_runner.query(
                fasta_path
            )[0]
            uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(
                jackhmmer_uniref90_result["sto"],
                max_sequences=self.uniref_max_hits
            )
            uniref90_out_path = os.path.join(output_dir, "uniref90_hits.a3m")
            with open(uniref90_out_path, "w") as f:
                f.write(uniref90_msa_as_a3m)

            if (self.hhsearch_pdb70_runner is not None):
                hhsearch_result = self.hhsearch_pdb70_runner.query(
                    uniref90_msa_as_a3m
                )
                pdb70_out_path = os.path.join(output_dir, "pdb70_hits.hhr")
                with open(pdb70_out_path, "w") as f:
                    f.write(hhsearch_result)

        if (self.jackhmmer_mgnify_runner is not None):
            jackhmmer_mgnify_result = self.jackhmmer_mgnify_runner.query(
                fasta_path
            )[0]
            mgnify_msa_as_a3m = parsers.convert_stockholm_to_a3m(
                jackhmmer_mgnify_result["sto"],
                max_sequences=self.mgnify_max_hits
            )
            mgnify_out_path = os.path.join(output_dir, "mgnify_hits.a3m")
            with open(mgnify_out_path, "w") as f:
                f.write(mgnify_msa_as_a3m)

        if (self.use_small_bfd and self.jackhmmer_small_bfd_runner is not None):
            jackhmmer_small_bfd_result = self.jackhmmer_small_bfd_runner.query(
                fasta_path
            )[0]
            bfd_out_path = os.path.join(output_dir, "small_bfd_hits.sto")
            with open(bfd_out_path, "w") as f:
                f.write(jackhmmer_small_bfd_result["sto"])
        elif (self.hhblits_bfd_uniclust_runner is not None):
            hhblits_bfd_uniclust_result = (
                self.hhblits_bfd_uniclust_runner.query(fasta_path)
            )
            if output_dir is not None:
                bfd_out_path = os.path.join(output_dir, "bfd_uniclust_hits.a3m")
                with open(bfd_out_path, "w") as f:
                    f.write(hhblits_bfd_uniclust_result["a3m"])


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


PICO_TO_ANGSTROM = 0.01


def from_proteinnet_string(proteinnet_str: str) -> Protein:
    tag_re = r'(\[[A-Z]+\]\n)'
    tags = [
        tag.strip() for tag in re.split(tag_re, proteinnet_str) if len(tag) > 0
    ]
    groups = zip(tags[0::2], [l.split('\n') for l in tags[1::2]])

    atoms = ['N', 'CA', 'C']
    aatype = None
    atom_positions = None
    atom_mask = None
    for g in groups:
        if ("[PRIMARY]" == g[0]):
            seq = g[1][0].strip()
            for i in range(len(seq)):
                if (seq[i] not in residue_constants.restypes):
                    seq[i] = 'X'
            aatype = np.array([
                residue_constants.restype_order.get(
                    res_symbol, residue_constants.restype_num
                ) for res_symbol in seq
            ])
        elif ("[TERTIARY]" == g[0]):
            tertiary = []
            for axis in range(3):
                tertiary.append(list(map(float, g[1][axis].split())))
            tertiary_np = np.array(tertiary)
            atom_positions = np.zeros(
                (len(tertiary[0]) // 3, residue_constants.atom_type_num, 3)
            ).astype(np.float32)
            for i, atom in enumerate(atoms):
                atom_positions[:, residue_constants.atom_order[atom], :] = (
                    np.transpose(tertiary_np[:, i::3])
                )
            atom_positions *= PICO_TO_ANGSTROM
        elif ("[MASK]" == g[0]):
            mask = np.array(list(map({'-': 0, '+': 1}.get, g[1][0].strip())))
            atom_mask = np.zeros(
                (len(mask), residue_constants.atom_type_num,)
            ).astype(np.float32)
            for i, atom in enumerate(atoms):
                atom_mask[:, residue_constants.atom_order[atom]] = 1
            atom_mask *= mask[..., None]

    return Protein(
        atom_positions=atom_positions,
        atom_mask=atom_mask,
        aatype=aatype,
        residue_index=np.arange(len(aatype)),
        b_factors=None,
    )


class DataPipeline:
    """Assembles input features."""

    def __init__(
            self,
            template_featurizer: Optional[templates.TemplateHitFeaturizer],
    ):
        self.template_featurizer = template_featurizer

    def _parse_msa_data(
            self,
            alignment_dir: str,
            alignment_index: Optional[Any] = None,
    ) -> Mapping[str, Any]:
        msa_data = {}
        if (alignment_index is not None):
            fp = open(os.path.join(alignment_dir, alignment_index["db"]), "rb")

            def read_msa(start, size):
                fp.seek(start)
                msa = fp.read(size).decode("utf-8")
                return msa

            for (name, start, size) in alignment_index["files"]:
                ext = os.path.splitext(name)[-1]

                if (ext == ".a3m"):
                    msa, deletion_matrix = parsers.parse_a3m(
                        read_msa(start, size)
                    )
                    data = {"msa": msa, "deletion_matrix": deletion_matrix}
                elif (ext == ".sto"):
                    msa, deletion_matrix, _ = parsers.parse_stockholm(
                        read_msa(start, size)
                    )
                    data = {"msa": msa, "deletion_matrix": deletion_matrix}
                else:
                    continue

                msa_data[name] = data

            fp.close()
        else:
            for f in os.listdir(alignment_dir):
                path = os.path.join(alignment_dir, f)
                ext = os.path.splitext(f)[-1]

                if (ext == ".a3m"):
                    with open(path, "r") as fp:
                        msa, deletion_matrix = parsers.parse_a3m(fp.read())
                    data = {"msa": msa, "deletion_matrix": deletion_matrix}
                elif (ext == ".sto"):
                    with open(path, "r") as fp:
                        msa, deletion_matrix, _ = parsers.parse_stockholm(
                            fp.read()
                        )
                    data = {"msa": msa, "deletion_matrix": deletion_matrix}
                else:
                    continue

                msa_data[f] = data

        return msa_data

    def _parse_template_hits(
            self,
            alignment_dir: str,
            alignment_index: Optional[Any] = None
    ) -> Mapping[str, Any]:
        all_hits = {}
        if (alignment_index is not None):
            fp = open(os.path.join(alignment_dir, alignment_index["db"]), 'rb')

            def read_template(start, size):
                fp.seek(start)
                return fp.read(size).decode("utf-8")

            for (name, start, size) in alignment_index["files"]:
                ext = os.path.splitext(name)[-1]

                if (ext == ".hhr"):
                    hits = parsers.parse_hhr(read_template(start, size))
                    all_hits[name] = hits

            fp.close()
        else:
            for f in os.listdir(alignment_dir):
                path = os.path.join(alignment_dir, f)
                ext = os.path.splitext(f)[-1]

                if (ext == ".hhr"):
                    with open(path, "r") as fp:
                        hits = parsers.parse_hhr(fp.read())
                    all_hits[f] = hits

        return all_hits

    def _get_msas(self,
                  alignment_dir: str,
                  input_sequence: Optional[str] = None,
                  alignment_index: Optional[str] = None,
                  ):
        msa_data = self._parse_msa_data(alignment_dir, alignment_index)
        if (len(msa_data) == 0):
            if (input_sequence is None):
                raise ValueError(
                    """
                    If the alignment dir contains no MSAs, an input sequence 
                    must be provided.
                    """
                )
            msa_data["dummy"] = {
                "msa": [input_sequence],
                "deletion_matrix": [[0 for _ in input_sequence]],
            }

        msas, deletion_matrices = zip(*[
            (v["msa"], v["deletion_matrix"]) for v in msa_data.values()
        ])

        return msas, deletion_matrices

    def _process_msa_feats(
            self,
            alignment_dir: str,
            input_sequence: Optional[str] = None,
            alignment_index: Optional[str] = None
    ) -> Mapping[str, Any]:
        msas, deletion_matrices = self._get_msas(
            alignment_dir, input_sequence, alignment_index
        )
        msa_features = make_msa_features(
            msas=msas,
            deletion_matrices=deletion_matrices,
        )

        return msa_features

    def process_fasta(
            self,
            fasta_path: str,
            alignment_dir: str,
            alignment_index: Optional[str] = None,
    ) -> FeatureDict:
        """Assembles features for a single sequence in a FASTA file"""
        with open(fasta_path) as f:
            fasta_str = f.read()
        input_seqs, input_descs = parsers.parse_fasta(fasta_str)
        if len(input_seqs) != 1:
            raise ValueError(
                f"More than one input sequence found in {fasta_path}."
            )
        input_sequence = input_seqs[0]
        input_description = input_descs[0]
        num_res = len(input_sequence)

        hits = self._parse_template_hits(alignment_dir, alignment_index)
        template_features = make_template_features(
            input_sequence,
            hits,
            self.template_featurizer,
        )

        sequence_features = make_sequence_features(
            sequence=input_sequence,
            description=input_description,
            num_res=num_res,
        )

        msa_features = self._process_msa_feats(alignment_dir, input_sequence, alignment_index)

        return {
            **sequence_features,
            **msa_features,
            **template_features
        }

    def process_mmcif(
            self,
            mmcif: mmcif_parsing.MmcifObject,  # parsing is expensive, so no path
            alignment_dir: str,
            chain_id: Optional[str] = None,
            alignment_index: Optional[str] = None,
    ) -> FeatureDict:
        """
            Assembles features for a specific chain in an mmCIF object.

            If chain_id is None, it is assumed that there is only one chain
            in the object. Otherwise, a ValueError is thrown.
        """
        if chain_id is None:
            chains = mmcif.structure.get_chains()
            chain = next(chains, None)
            if chain is None:
                raise ValueError("No chains in mmCIF file")
            chain_id = chain.id

        mmcif_feats = make_mmcif_features(mmcif, chain_id)

        input_sequence = mmcif.chain_to_seqres[chain_id]
        hits = self._parse_template_hits(alignment_dir, alignment_index)
        template_features = make_template_features(
            input_sequence,
            hits,
            self.template_featurizer,
            query_release_date=to_date(mmcif.header["release_date"])
        )

        msa_features = self._process_msa_feats(alignment_dir, input_sequence, alignment_index)

        return {**mmcif_feats, **template_features, **msa_features}

    def process_pdb(
            self,
            pdb_path: str,
            alignment_dir: str,
            is_distillation: bool = True,
            chain_id: Optional[str] = None,
            _structure_index: Optional[str] = None,
            alignment_index: Optional[str] = None,
    ) -> FeatureDict:
        """
            Assembles features for a protein in a PDB file.
        """
        if (_structure_index is not None):
            db_dir = os.path.dirname(pdb_path)
            db = _structure_index["db"]
            db_path = os.path.join(db_dir, db)
            fp = open(db_path, "rb")
            _, offset, length = _structure_index["files"][0]
            fp.seek(offset)
            pdb_str = fp.read(length).decode("utf-8")
            fp.close()
        else:
            with open(pdb_path, 'r') as f:
                pdb_str = f.read()

        protein_object = from_pdb_string(pdb_str, chain_id)
        input_sequence = _aatype_to_str_sequence(protein_object.aatype)
        description = os.path.splitext(os.path.basename(pdb_path))[0].upper()
        pdb_feats = make_pdb_features(
            protein_object,
            description,
            is_distillation=is_distillation
        )

        hits = self._parse_template_hits(alignment_dir, alignment_index)
        template_features = make_template_features(
            input_sequence,
            hits,
            self.template_featurizer,
        )

        msa_features = self._process_msa_feats(alignment_dir, input_sequence, alignment_index)

        return {**pdb_feats, **template_features, **msa_features}

    def process_core(
            self,
            core_path: str,
            alignment_dir: str,
            alignment_index: Optional[str] = None,
    ) -> FeatureDict:
        """
            Assembles features for a protein in a ProteinNet .core file.
        """
        with open(core_path, 'r') as f:
            core_str = f.read()

        protein_object = from_proteinnet_string(core_str)
        input_sequence = _aatype_to_str_sequence(protein_object.aatype)
        description = os.path.splitext(os.path.basename(core_path))[0].upper()
        core_feats = make_protein_features(protein_object, description)

        hits = self._parse_template_hits(alignment_dir, alignment_index)
        template_features = make_template_features(
            input_sequence,
            hits,
            self.template_featurizer,
        )

        msa_features = self._process_msa_feats(alignment_dir, input_sequence)

        return {**core_feats, **template_features, **msa_features}

    def process_multiseq_fasta(self,
                               fasta_path: str,
                               super_alignment_dir: str,
                               ri_gap: int = 200,
                               ) -> FeatureDict:
        """
            Assembles features for a multi-sequence FASTA. Uses Minkyung Baek's
            hack from Twitter (a.k.a. AlphaFold-Gap).
        """
        with open(fasta_path, 'r') as f:
            fasta_str = f.read()

        input_seqs, input_descs = parsers.parse_fasta(fasta_str)

        # No whitespace allowed
        input_descs = [i.split()[0] for i in input_descs]

        # Stitch all of the sequences together
        input_sequence = ''.join(input_seqs)
        input_description = '-'.join(input_descs)
        num_res = len(input_sequence)

        sequence_features = make_sequence_features(
            sequence=input_sequence,
            description=input_description,
            num_res=num_res,
        )

        seq_lens = [len(s) for s in input_seqs]
        total_offset = 0
        for sl in seq_lens:
            total_offset += sl
            sequence_features["residue_index"][total_offset:] += ri_gap

        msa_list = []
        deletion_mat_list = []
        for seq, desc in zip(input_seqs, input_descs):
            alignment_dir = os.path.join(
                super_alignment_dir, desc
            )
            msas, deletion_mats = self._get_msas(
                alignment_dir, seq, None
            )
            msa_list.append(msas)
            deletion_mat_list.append(deletion_mats)

        final_msa = []
        final_deletion_mat = []
        msa_it = enumerate(zip(msa_list, deletion_mat_list))
        for i, (msas, deletion_mats) in msa_it:
            prec, post = sum(seq_lens[:i]), sum(seq_lens[i + 1:])
            msas = [
                [prec * '-' + seq + post * '-' for seq in msa] for msa in msas
            ]
            deletion_mats = [
                [prec * [0] + dml + post * [0] for dml in deletion_mat]
                for deletion_mat in deletion_mats
            ]

            assert (len(msas[0][-1]) == len(input_sequence))

            final_msa.extend(msas)
            final_deletion_mat.extend(deletion_mats)

        msa_features = make_msa_features(
            msas=final_msa,
            deletion_matrices=final_deletion_mat,
        )

        template_feature_list = []
        for seq, desc in zip(input_seqs, input_descs):
            alignment_dir = os.path.join(
                super_alignment_dir, desc
            )
            hits = self._parse_template_hits(alignment_dir, alignment_index=None)
            template_features = make_template_features(
                seq,
                hits,
                self.template_featurizer,
            )
            template_feature_list.append(template_features)

        template_features = unify_template_features(template_feature_list)

        return {
            **sequence_features,
            **msa_features,
            **template_features,
        }
