import os
from functools import lru_cache

import mdtraj as md
import numpy as np


# ---------------------------------------------------------------------------
#  Terminal sidechain atom definitions
# ---------------------------------------------------------------------------

TERMINAL_SC_ATOM = {
    "ALA": "CB",
    "ARG": "NH1",
    "ASN": "OD1",
    "ASP": "OD1",
    "CYS": "SG",
    "CYM": "SG",
    "GLU": "OE1",
    "GLN": "OE1",
    "GLY": None,
    "HIS": "NE2",
    "ILE": "CD1",
    "LEU": "CD1",
    "LYS": "NZ",
    "MET": "CE",
    "PHE": "CZ",
    "PRO": "CG",
    "SER": "OG",
    "THR": "OG1",
    "TRP": "CH2",
    "TYR": "OH",
    "VAL": "CG1",
}

_BB_NAMES = ("N", "CA", "C", "O")
_ANG_PER_NM = 10.0
_CB_BOND_ANG = 1.522  # CA-CB bond length in Angstroms


def _virtual_cb(n, ca, c):
    """Compute virtual CB from backbone N, CA, C (coordinates in nm).

    Uses the standard tetrahedral-geometry formula with coefficients from
    Ingraham et al. 2019.  Internally converts to Angstroms so the
    coefficients remain valid, then converts back to nm.
    """
    n_a, ca_a, c_a = n * _ANG_PER_NM, ca * _ANG_PER_NM, c * _ANG_PER_NM
    b = ca_a - n_a
    cv = c_a - ca_a
    a = np.cross(b, cv)
    cb_vec = -0.58273431 * a + 0.56802827 * b - 0.54067466 * cv
    nrm = np.linalg.norm(cb_vec)
    if nrm > 1e-8:
        cb_vec = cb_vec / nrm * _CB_BOND_ANG
    return (ca_a + cb_vec) / _ANG_PER_NM


def _extract_5atom_xyz(struc, frame=0):
    """Extract ``[N, CA, C, O, T]`` coordinates for every protein residue.

    *T* is the terminal (furthest-from-CA) sidechain heavy atom.
    For Glycine a virtual CB is placed using tetrahedral geometry.

    Parameters
    ----------
    struc : mdtraj.Trajectory
        Must contain **all protein atoms** (not backbone-only).
    frame : int
        Frame index to read coordinates from.

    Returns
    -------
    coords : np.ndarray, shape ``[n_residues, 5, 3]``
    """
    xyz = struc.xyz[frame]
    top = struc.top
    n_res = top.n_residues
    coords = np.zeros([n_res, 5, 3], dtype=np.float32)

    for res in top.residues:
        ri = res.index
        atom_pos = {}
        sc_heavy = []
        for a in res.atoms:
            atom_pos[a.name] = xyz[a.index]
            if a.name not in _BB_NAMES and a.element.symbol != "H":
                sc_heavy.append(xyz[a.index])

        for j, name in enumerate(_BB_NAMES):
            if name in atom_pos:
                coords[ri, j] = atom_pos[name]

        target = TERMINAL_SC_ATOM.get(res.name)
        if target is None:
            # Glycine or unmapped residue – use virtual CB
            if all(n in atom_pos for n in _BB_NAMES[:3]):
                coords[ri, 4] = _virtual_cb(
                    atom_pos["N"], atom_pos["CA"], atom_pos["C"]
                )
            elif "CA" in atom_pos:
                coords[ri, 4] = atom_pos["CA"]
        elif target in atom_pos:
            coords[ri, 4] = atom_pos[target]
        else:
            # Fallback: furthest sidechain heavy atom from CA
            ca = atom_pos.get("CA")
            if ca is not None and sc_heavy:
                sc_arr = np.array(sc_heavy)
                dists = np.linalg.norm(sc_arr - ca, axis=-1)
                coords[ri, 4] = sc_arr[np.argmax(dists)]
            elif ca is not None and all(n in atom_pos for n in _BB_NAMES[:3]):
                coords[ri, 4] = _virtual_cb(
                    atom_pos["N"], ca, atom_pos["C"]
                )
            elif ca is not None:
                coords[ri, 4] = ca

    return coords


# ---------------------------------------------------------------------------
#  Amino-acid lookup tables
# ---------------------------------------------------------------------------

abbrev = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "CYM": "C",
    "GLU": "E",
    "GLN": "Q",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
lookup = {
    "C": 4,
    "D": 3,
    "S": 15,
    "Q": 5,
    "K": 11,
    "I": 9,
    "P": 14,
    "T": 16,
    "F": 13,
    "A": 0,
    "G": 7,
    "H": 8,
    "E": 6,
    "L": 10,
    "R": 1,
    "W": 17,
    "V": 19,
    "N": 2,
    "Y": 18,
    "M": 12,
}


def resolve_local_path(original_path, structure_dir):
    candidate = os.path.join(structure_dir, os.path.basename(original_path))
    if os.path.exists(candidate):
        return candidate
    if os.path.exists(original_path):
        return original_path
    raise FileNotFoundError(
        f"Could not resolve '{original_path}'. Expected a local copy under '{structure_dir}'."
    )


def load_task2_entries(data_dir, filestem, dataset_subdir="task2"):
    x_path = os.path.join(data_dir, dataset_subdir, f"X-train-{filestem}.npy")
    y_path = os.path.join(data_dir, dataset_subdir, f"y-train-{filestem}.npy")
    X_train = np.load(x_path, allow_pickle=True)
    y_train = np.load(y_path, allow_pickle=True)
    return list(zip(X_train, y_train))


def determine_global_weights(data_dir, filestem, positive_cutoff, negative_cutoff, dataset_subdir="task2"):
    y_train = np.load(
        os.path.join(data_dir, dataset_subdir, f"y-train-{filestem}.npy"), allow_pickle=True
    )
    all_examples = np.concatenate(y_train)
    valid_examples = all_examples[all_examples >= 0]
    number_positive_examples = np.sum(valid_examples >= positive_cutoff)
    number_negative_examples = np.sum(valid_examples < negative_cutoff)
    total_examples = number_positive_examples + number_negative_examples
    positive_weight = 1 / number_positive_examples * (total_examples / 2.0)
    negative_weight = 1 / number_negative_examples * (total_examples / 2.0)
    return positive_weight, negative_weight


@lru_cache(maxsize=256)
def _load_pdb_backbone(pdb_path):
    pdb = md.load(pdb_path)
    prot_iis = pdb.top.select("protein and (name N or name CA or name C or name O)")
    return pdb.atom_slice(prot_iis)


_UNKNOWN_AATYPE = 20   # AF2 convention for non-standard residues


def sequence_to_ids(pdb):
    seq = [r.name for r in pdb.top.residues]
    return np.asarray(
        [lookup.get(abbrev.get(a), _UNKNOWN_AATYPE) for a in seq],
        dtype=np.int32,
    )


def process_strucs(strucs, num_atoms=4):
    bb_pdbs = []
    full_prots = [] if num_atoms == 5 else None
    for s in strucs:
        prot_iis = s.top.select("protein and (name N or name CA or name C or name O)")
        prot_bb = s.atom_slice(prot_iis)
        bb_pdbs.append(prot_bb)
        if num_atoms == 5:
            all_prot_iis = s.top.select("protein")
            full_prots.append(s.atom_slice(all_prot_iis))

    B = len(strucs)
    L_max = np.max([pdb.top.n_residues for pdb in bb_pdbs])
    X = np.zeros([B, L_max, num_atoms, 3], dtype=np.float32)
    S = np.zeros([B, L_max], dtype=np.int32)

    for i, prot_bb in enumerate(bb_pdbs):
        l = prot_bb.top.n_residues
        if num_atoms == 5:
            xyz = _extract_5atom_xyz(full_prots[i])
        else:
            xyz = prot_bb.xyz.reshape(l, 4, 3)
        S[i, :l] = sequence_to_ids(prot_bb)
        X[i] = np.pad(xyz, [[0, L_max - l], [0, 0], [0, 0]], "constant", constant_values=(np.nan,))

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.0
    X = np.nan_to_num(X)
    return X, S, mask


def process_pdb_paths(pdb_paths, num_atoms=4):
    return process_strucs([md.load(path) for path in pdb_paths], num_atoms=num_atoms)


def parse_batch(batch, structure_dir, num_atoms=4, y_dtype="float32"):
    pdbs = []
    for x, _ in batch:
        _, pdb_fn, _ = x
        pdb_path = resolve_local_path(pdb_fn, structure_dir)
        pdbs.append(_load_pdb_backbone(pdb_path))

    B = len(batch)
    L_max = np.max([pdb.top.n_residues for pdb in pdbs])
    X = np.zeros([B, L_max, num_atoms, 3], dtype=np.float32)
    S = np.zeros([B, L_max], dtype=np.int32)
    y = np.zeros([B, L_max], dtype=np.float32 if y_dtype == "float32" else np.int32) - 1
    meta = []

    for i, (x, targs) in enumerate(batch):
        traj_fn, pdb_fn, traj_iis = x
        traj_path = resolve_local_path(traj_fn, structure_dir)
        pdb_path = resolve_local_path(pdb_fn, structure_dir)
        traj_iis = int(traj_iis)

        struc = md.load_frame(traj_path, traj_iis, top=pdb_path)

        pdb = pdbs[i]
        l = pdb.top.n_residues

        if num_atoms == 5:
            prot_iis = struc.top.select("protein")
            prot = struc.atom_slice(prot_iis)
            xyz = _extract_5atom_xyz(prot)
        else:
            prot_iis = struc.top.select(
                "protein and (name N or name CA or name C or name O)"
            )
            prot_bb = struc.atom_slice(prot_iis)
            xyz = prot_bb.xyz.reshape(l, 4, 3)

        S[i, :l] = sequence_to_ids(pdb)
        X[i] = np.pad(xyz, [[0, L_max - l], [0, 0], [0, 0]], "constant", constant_values=(np.nan,))
        y[i, :l] = targs
        meta.append((traj_path, pdb_path, traj_iis))

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.0
    X = np.nan_to_num(X)
    return X, S, y, np.asarray(meta, dtype=str), mask


class DynamicLoader:
    def __init__(self, dataset, structure_dir, batch_size=32, shuffle=True,
                 num_atoms=4, y_dtype="float32"):
        self.dataset = dataset
        self.structure_dir = structure_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_atoms = num_atoms
        self.y_dtype = y_dtype

    def chunks(self, arr, chunk_size):
        for i in range(0, len(arr), chunk_size):
            yield arr[i : i + chunk_size]

    def batch(self):
        dataset = list(self.dataset)
        if self.shuffle:
            np.random.shuffle(dataset)
        self.clusters = list(self.chunks(dataset, self.batch_size))

    def __iter__(self):
        self.batch()
        if self.shuffle:
            np.random.shuffle(self.clusters)
        for batch in self.clusters:
            yield parse_batch(
                batch,
                structure_dir=self.structure_dir,
                num_atoms=self.num_atoms,
                y_dtype=self.y_dtype,
            )


def simulation_dataset(data_dir, structure_dir, batch_size, filestem,
                        dataset_subdir="task2", shuffle=True, num_atoms=4):
    trainset = load_task2_entries(data_dir, filestem, dataset_subdir=dataset_subdir)
    loader = DynamicLoader(
        trainset, structure_dir=structure_dir, batch_size=batch_size,
        shuffle=shuffle, num_atoms=num_atoms,
    )
    return loader


def load_label_dictionary(label_dict_path):
    return np.load(label_dict_path, allow_pickle=True).item()


def process_apo_ids(apo_ids, apo_structure_dir, num_atoms=4):
    """Process apo crystal structures from their PDB IDs.

    Looks for PDB files named ``{apo_id}_clean_h.pdb`` inside
    *apo_structure_dir*.  Returns featurized arrays suitable for
    model input.
    """
    pdb_paths = []
    for apo_id in apo_ids:
        candidate = os.path.join(apo_structure_dir, f"{apo_id}_clean_h.pdb")
        if not os.path.exists(candidate):
            candidate = os.path.join(apo_structure_dir, f"{apo_id.upper()}_clean_h.pdb")
        if not os.path.exists(candidate):
            candidate = os.path.join(apo_structure_dir, f"{apo_id.lower()}_clean_h.pdb")
        if not os.path.exists(candidate):
            raise FileNotFoundError(
                f"Cannot find PDB for apo ID '{apo_id}' in '{apo_structure_dir}'"
            )
        pdb_paths.append(candidate)

    strucs = [md.load(p) for p in pdb_paths]
    return process_strucs(strucs, num_atoms=num_atoms)


# ---------------------------------------------------------------------------
#  Precomputed feature cache & fast in-memory loader
# ---------------------------------------------------------------------------

import pickle as _pickle


def precompute_all_samples(data_dir, structure_dir, filestem,
                           dataset_subdir="task2", num_atoms=4,
                           cache_path=None):
    """Pre-extract coordinate features for every training sample.

    If *cache_path* exists the cached data is loaded directly, skipping all
    trajectory / PDB I/O.  Otherwise features are computed from the raw files
    and persisted to *cache_path* for future runs.

    Returns
    -------
    list[dict]
        Each element has keys ``xyz`` (ndarray ``[L, A, 3]``),
        ``seq`` (ndarray ``[L]``), and ``y`` (ndarray ``[L]``).
    """
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as fh:
                data = _pickle.load(fh)
            print(f"Loaded precomputed features: {cache_path} "
                  f"({len(data)} samples)")
            print(f"  (delete this file to force re-extraction)")
            return data
        except Exception as exc:
            print(f"WARNING: cache load failed ({exc}), recomputing …")

    entries = load_task2_entries(data_dir, filestem, dataset_subdir)
    n_total = len(entries)
    print(f"Precomputing features for {n_total} samples "
          f"({num_atoms}-atom) …")

    samples = []
    for idx, (x, y_labels) in enumerate(entries):
        traj_fn, pdb_fn, traj_iis = x
        traj_path = resolve_local_path(traj_fn, structure_dir)
        pdb_path = resolve_local_path(pdb_fn, structure_dir)
        traj_iis_int = int(traj_iis)

        pdb = _load_pdb_backbone(pdb_path)
        struc = md.load_frame(traj_path, traj_iis_int, top=pdb_path)
        l = pdb.top.n_residues

        if num_atoms == 5:
            prot_iis = struc.top.select("protein")
            prot = struc.atom_slice(prot_iis)
            xyz = _extract_5atom_xyz(prot)
        else:
            prot_iis = struc.top.select(
                "protein and (name N or name CA or name C or name O)"
            )
            prot_bb = struc.atom_slice(prot_iis)
            xyz = prot_bb.xyz.reshape(l, num_atoms, 3)

        seq_ids = sequence_to_ids(pdb)
        samples.append({
            "xyz": xyz.astype(np.float32),
            "seq": seq_ids.astype(np.int32),
            "y": np.asarray(y_labels, dtype=np.float32),
        })
        if (idx + 1) % 500 == 0 or idx + 1 == n_total:
            print(f"  {idx + 1} / {n_total}")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as fh:
            _pickle.dump(samples, fh, protocol=_pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(cache_path) / 1024 / 1024
        print(f"  Cached to {cache_path} ({size_mb:.1f} MB)")

    return samples


class PrecomputedLoader:
    """Drop-in replacement for *DynamicLoader* using pre-extracted features.

    Produces batches in the same ``(X, S, y, meta, mask)`` format as
    :func:`parse_batch` so existing training loops work unchanged.
    """

    def __init__(self, samples, batch_size=4, shuffle=True):
        self.samples = samples
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __iter__(self):
        indices = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(indices)

        chunks = []
        for start in range(0, len(indices), self.batch_size):
            chunks.append(indices[start : start + self.batch_size])
        if self.shuffle:
            np.random.shuffle(chunks)

        for chunk in chunks:
            batch = [self.samples[int(i)] for i in chunk]
            yield self._collate(batch)

    @staticmethod
    def _collate(batch):
        B = len(batch)
        L_max = max(s["xyz"].shape[0] for s in batch)
        num_atoms = batch[0]["xyz"].shape[1]

        X = np.zeros([B, L_max, num_atoms, 3], dtype=np.float32)
        S = np.zeros([B, L_max], dtype=np.int32)
        y = np.full([B, L_max], -1, dtype=np.float32)

        for i, s in enumerate(batch):
            l = s["xyz"].shape[0]
            X[i] = np.pad(
                s["xyz"], [[0, L_max - l], [0, 0], [0, 0]],
                "constant", constant_values=(np.nan,),
            )
            S[i, :l] = s["seq"]
            y[i, :l] = s["y"]

        isnan = np.isnan(X)
        mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
        X[isnan] = 0.0
        X = np.nan_to_num(X)

        return X, S, y, None, mask
