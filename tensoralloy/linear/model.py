#!coding=utf-8
import h5py
import numpy as np
import os.path
import joblib
from pathlib import Path
from ase.atoms import Atoms
from ase.neighborlist import neighbor_list
from ase.data import atomic_masses, atomic_numbers
from ase.units import GPa
from ase.calculators.calculator import Calculator, all_changes
from collections import Counter, namedtuple
from typing import List
from functools import partial
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.model_selection import ParameterGrid
from numpy import einsum as contract

from tensoralloy.linear.ops import setup_tensors, kernel_F1, kernel_F2, sum_forces, \
                                   sum_dG, fill_tensors

filter_presets = {
    "pexp": {
        "small": {
            "rl": np.linspace(1.0, 4.0, num=8, endpoint=True),
            "pl": np.linspace(3.0, 1.0, num=8, endpoint=True)
        },
        "medium": {
            "rl": np.linspace(1.0, 4.0, num=16, endpoint=True),
            "pl": np.linspace(3.0, 1.0, num=16, endpoint=True)
        },
        "large": {
            "rl": np.linspace(1.0, 4.0, num=32, endpoint=True),
            "pl": np.linspace(3.0, 1.0, num=32, endpoint=True)
        }
    },
    "morse": {
        "small": {
            "D": np.ones(1, dtype=float),
            "r0": np.linspace(1.4, 3.2, num=10, endpoint=True),
            "gamma": np.ones(1, dtype=float)
        },
        "medium": {
            "D": np.ones(1, dtype=float),
            "r0": np.linspace(1.4, 3.2, num=10, endpoint=True),
            "gamma": np.array([1.0, 2.0])
        },
        "large": {
            "D": np.array([0.8, 1.2]),
            "r0": np.linspace(1.4, 3.2, num=10, endpoint=True),
            "gamma": np.array([1.0, 2.0])
        }
    }
}


def cantor_pairing(x, y):
    """
    The Cantor Pairing function:

        f(x, y) = (x + y)(x + y + 1) // 2 + y

    f(x, y) will only be unique if x and y are integers.

    See Also
    --------
    https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function

    """
    x = np.asarray(x)
    y = np.asarray(y)
    return (x + y) * (x + y + 1) // 2 + y


def get_T_dm(max_moment, symmetric=False):
    """
    Return the rank-2 multiplicity tensor T_dm.
    """
    dims = [1, 4, 10, 20, 35, 56, 84]
    if max_moment > 6:
        raise ValueError("The maximum angular moment should be <= 6")
    dmax = dims[max_moment]
    array = np.zeros((dmax, max_moment + 1), dtype=np.float64)
    array[0, 0] = 1
    if max_moment > 0:
        array[1: 4, 1] = 1
    if max_moment > 1:
        array[4: 10, 2] = 1, 2, 2, 1, 2, 1
        if symmetric:
            array[0, 2] = -1 / 3
    if max_moment > 2:
        array[10: 20, 3] = 1, 3, 3, 3, 6, 3, 1, 3, 3, 1
        if symmetric:
            array[1: 4, 3] = -3 / 5
    if max_moment > 3:
        array[20: 35, 4] = 1, 4, 4, 6, 12, 6, 4, 12, 12, 4, 1, 4, 6, 4, 1
    if max_moment > 4:
        array[35: 56, 5] = 1, 5, 5, 10, 20, 10, 10, 30, 30, 10, 5, 20, \
                           30, 20, 5, 1, 5, 10, 10, 5, 1
    if max_moment > 5:
        array[56: 84, 6] = 1, 5, 5, 10, 20, 10, 10, 30, 30, 10, 5, 20, \
                           30, 20, 5, 1, 5, 10, 10, 5, 1
    return array


def fcut(x: np.ndarray, cutforce: float):
    """
    The cosine cutoff function.
    """
    z = np.minimum(1.0, x / cutforce) * np.pi
    f = 0.5 * (np.cos(z) + 1)
    df = -0.5 * np.pi / cutforce * np.sin(z)
    return f, df


def pexp(x: np.ndarray, r: float, p: float):
    """
    The parameterized exponential function.
    """
    f = np.exp(-(x / r) ** p)
    df = -f * p * (x / r) ** (p - 1.0) / r
    return f, df


def morse(x: np.ndarray, D, gamma, r0):
    """
    The morse function.
    f(x) = d * [ exp(-2 * gamma * (r - r0)) - 2 * exp(-gamma * (r - r0)) ]
    """
    dr = gamma * (x - r0)
    f = D * (np.exp(-2 * dr) - 2 * np.exp(-dr))
    df = -D * gamma * 2 * (np.exp(-2 * dr) - np.exp(-dr))
    return f, df


def get_filter_preset(key: str):
    """
    Return a pre-defined collection of filter functions.
    """
    vals = key.split("@")
    if len(vals) != 2:
        raise KeyError(f"{key} is not valid preset. Should be func@size")
    func = vals[0]
    size = vals[1]
    if func == "pexp":
        params = filter_presets[func][size]
        filters = [partial(pexp, r=params['rl'][i], p=params['pl'][i])
                   for i in range(len(params['rl']))]
    elif func == "morse":
        params = filter_presets[func][size]
        grid = ParameterGrid(params)
        filters = []
        for row in grid:
            D = row["D"]
            r0 = row["r0"]
            gamma = row["gamma"]
            filters.append(partial(morse, D=D, r0=r0, gamma=gamma))
    else:
        raise KeyError("Only morse or pexp functions are supported")
    return filters


def calculate_linear_mtp_parameters(atoms: Atoms, species: List[str], rcut=5.0,
                                    max_moment=3, filters='small',
                                    fill_tensors_op=fill_tensors,
                                    sum_dG_op=sum_dG):
    """
    Compute the linear moment tensor potential parameters.
    """
    i_l, j_l, D_lx, R_l = neighbor_list('ijDd', atoms, cutoff=rcut)
    T_dm = get_T_dm(max_moment)
    elements = atoms.get_chemical_symbols()

    # Setup the radial filter functions
    if isinstance(filters, str):
        filters = get_filter_preset(filters)

    # Setup dimensions
    ddim, mdim = T_dm.shape
    adim = len(atoms)
    bdim = len(species)
    nnl = 0
    for i in range(len(atoms)):
        indices = np.where(i_l == i)[0]
        ii = atoms.numbers[i_l[indices]]
        ij = atoms.numbers[i_l[indices]]
        if len(ii) > 0:
            nnl = max(max(Counter(cantor_pairing(ii, ij)).values()), nnl)
    cdim = nnl
    kdim = len(filters)
    xdim = 3

    # Setup istart for each specie if `reorder` is True
    eltnum = Counter(atoms.get_chemical_symbols())
    istart = Counter()
    offset = 0
    for specie in species:
        istart[specie] = offset
        offset += eltnum[specie]
    loc = np.zeros(adim, dtype=np.int32)
    reordered = False
    for i, specie in enumerate(atoms.get_chemical_symbols()):
        loc[i] = istart[specie]
        if loc[i] != i:
            reordered = True
        istart[specie] += 1

    # Compute H_kl
    H_lk = np.zeros((len(i_l), kdim), dtype=np.float64)
    dH_lk = np.zeros_like(H_lk)
    s_l, ds_l = fcut(R_l, cutforce=rcut)
    for k in range(kdim):
        fk, dfk = filters[k](R_l)
        H_lk[:, k] = fk * s_l
        dH_lk[:, k] = fk * ds_l + dfk * s_l

    # Setup integer arrays: `eltypes` and `eltmap`.
    eltypes = np.zeros(adim, dtype=np.int32)
    for i in range(adim):
        eltypes[i] = species.index(elements[i])
    eltmap = np.zeros((bdim, bdim), dtype=np.int32)
    for i in range(bdim):
        k = 1
        for j in range(bdim):
            if i == j:
                eltmap[i, j] = 0
            else:
                eltmap[i, j] = k
                k += 1

    # Initialized tensors
    R_abc = np.zeros((adim, bdim, cdim), dtype=np.float64)
    drdrx_abcx = np.zeros((adim, bdim, cdim, xdim))
    H_abck = np.zeros((adim, bdim, cdim, kdim), dtype=np.float64)
    dHdr_abck = np.zeros_like(H_abck)
    M_abcd = np.zeros((adim, bdim, cdim, ddim), dtype=np.float64)
    dMdrx_abcdx = np.zeros((adim, bdim, cdim, ddim, xdim), dtype=np.float64)
    i_abc = np.zeros((adim, bdim, cdim), dtype=np.int32) - 1
    j_abc = np.zeros((adim, bdim, cdim), dtype=np.int32) - 1
    t_a = np.zeros(adim, dtype=np.int32)
    neigh = np.zeros((adim, bdim), dtype=np.int32)
    i_l = i_l.astype(np.int32)
    j_l = j_l.astype(np.int32)

    fill_tensors_op(eltmap, eltypes, i_l, j_l, neigh, loc, R_abc, R_l,
                    drdrx_abcx, D_lx, H_abck, H_lk, dHdr_abck, dH_lk, i_abc,
                    j_abc, t_a, M_abcd, dMdrx_abcdx, np.int32(max_moment))

    P_abkd = contract("abcd,abck->abkd", M_abcd, H_abck)
    sign = np.sign(P_abkd[:, :, :, 0])
    S_abkd = np.square(P_abkd)
    G_abkm = contract("dm,abkd->abkm", T_dm, S_abkd)
    G_abkm[:, :, :, 0] = np.sqrt(G_abkm[:, :, :, 0]) * sign

    dQ_abkm = np.ones_like(G_abkm)
    dQ_abkm[:, :, :, 0] = 0.5 / G_abkm[:, :, :, 0]
    dG1_abkmxc = contract("abkm,dm,abkd,abcd,abck,abcx->abkmxc",
                          dQ_abkm, T_dm, P_abkd, M_abcd, dHdr_abck, drdrx_abcx)
    dG2_abkmxc = contract("abkm,dm,abkd,abck,abcdx->abkmxc",
                          dQ_abkm, T_dm, P_abkd, H_abck, dMdrx_abcdx)
    dG_abkmxc = 2 * (dG1_abkmxc + dG2_abkmxc)
    dGdh_axybkm = np.zeros((adim, xdim, xdim, bdim, kdim, mdim))
    dGdrx_baxbkm = np.zeros((bdim, adim, xdim, bdim, kdim, mdim))

    sum_dG_op(np.int32(adim), np.int32(bdim), np.int32(cdim), np.int32(kdim),
              np.int32(mdim), i_abc, j_abc, t_a, dGdrx_baxbkm,
              dG_abkmxc, dGdh_axybkm, drdrx_abcx, R_abc)

    splits = np.cumsum([0] + [eltnum[specie] for specie in species])[1:-1]
    if len(species) > 1:
        G_abkm_array = np.split(G_abkm, splits, axis=0)
        dGdrx_baxbkm_array = np.split(dGdrx_baxbkm, len(species), axis=0)
        dGdh_axybkm_array = np.split(dGdh_axybkm, splits, axis=0)
    else:
        G_abkm_array = [G_abkm]
        dGdrx_baxbkm_array = [dGdrx_baxbkm]
        dGdh_axybkm_array = [dGdh_axybkm]

    MTNPResults = namedtuple(
        "MTNPResults",
        ("G_abkm", "dGdrx_axbkm", "dGdh_axybkm"))
    results = {}
    for i, specie in enumerate(species):
        results[specie] = MTNPResults(
            G_abkm_array[i], dGdrx_baxbkm_array[i][0], dGdh_axybkm_array[i])
    return results, loc, reordered


class LinearPotential(Calculator):
    """
    The dynamic linear moment tensor potential.

    >>> from tensoralloy.io.db import snap
    >>> from ase.build import bulk
    >>> db = snap("Ni")
    >>> lp = LinearPotential(["Ni"], "medium", 3, 6.0)
    >>> for atoms_id in range(1, len(db) + 1):
    >>>     lp.add(db.get_atoms(id=atoms_id, add_additional_information=True))
    >>> lp.fit()
    >>> print(lp.evaluate_mean_absolute_errors())
    >>> atoms = bulk("Ni")
    >>> print(lp.calculate(atoms))
    """

    # ASE Calculator properties
    implemented_properties = ["energy", "forces", "stress"]
    nolabel = True

    def __init__(self, species: List[str], preset: str, max_moment=3, rcut=5.0,
                 database="lp.hdf5", use_forces=True, use_stress=True,
                 max_db_size=16):
        """
        Initialization method.

        Parameters
        ----------
        species : List[str]
            A list of atom types.
        preset : str
            The radial descriptor filter preset. The format is type@size.
            Possible types are 'pexp' and 'morse'.
            Possible sizes are 'small', 'medium' and 'large'.
        max_moment : int
            The maximum angular moment. Defaults to 3. Maximum is 5.
        rcut : float
            The cutoff radius.
        database : str
            The database file name.
        use_forces : bool
            Use atomic forces to train the potential. True is recommended.
        use_stress : bool
            Use virial stress to train the potential. True is recommended.
        max_db_size : int
            The maximum size of the database in GB.

        """
        super(LinearPotential, self).__init__()

        self._species = sorted(species)
        self._eltmap = np.zeros((len(species), len(species)), dtype=int)
        for i in range(len(species)):
            k = 1
            for j in range(len(species)):
                if i == j:
                    self._eltmap[i, j] = 0
                else:
                    self._eltmap[i, j] = k
                    k += 1

        self._preset = preset
        self._filters = get_filter_preset(preset)
        self._max_moment = max_moment
        self._rcut = rcut
        self._c_bbkm = None
        self._use_forces = use_forces
        self._use_stress = use_stress
        self._bkm_shape = (len(species), len(self._filters), max_moment + 1)
        self._bkm_val = len(species) * (max_moment + 1) * len(self._filters)
        self._dim = len(species) * (self._bkm_val + 1)
        self._sig = self._get_signature()
        self._max_len = \
            1024**3 * max_db_size // (len(self._species) * self._bkm_val)

        self._h5_handle = None
        self._database_file = database
        self._ds_A = None
        self._ds_y = None
        self._ds_w = None
        self._ds_t = None

        self.open(database)

    def _get_signature(self):
        """
        Return the signature of this potential.
        """
        return f"{'-'.join(self._species)}_{self._max_moment}_" \
               f"{self._preset}_{self._rcut:.1f}" \
               f"_{int(self._use_forces)}_{int(self._use_stress)}"

    @property
    def solution(self):
        return self._h5_handle["x"][()]

    @classmethod
    def from_hdf5(cls, filename):
        """
        Initialization from an existing HDF5 database.
        """
        filename = Path(filename)
        if filename.exists():
            h5 = h5py.File(filename, mode='r+')
            sig = h5.get("sig", default=None)
            if sig is None:
                return None
            entries = sig[()]
            if isinstance(entries, bytes):
                entries = entries.decode("utf-8")
            entries = entries.split("_")
            h5.close()
            if len(entries) == 6:
                species = entries[0].split("-")
                max_moment = int(entries[1])
                preset = entries[2]
                rcut = float(entries[3])
                use_forces = bool(int(entries[4]))
                use_stress = bool(int(entries[5]))
                return cls(species=species, preset=preset,
                           max_moment=max_moment, rcut=rcut, database=filename,
                           use_forces=use_forces, use_stress=use_stress)
        return None

    def copy_hdf5(self, filename):
        """
        Copy A, y, row_type and row_weight from another hdf5 file.
        The signature must be the same.
        """
        filename = Path(filename)

        if filename.exists():
            h5 = h5py.File(filename, mode="r")
            sig = h5.get("sig", default=None)
            if sig and sig[()].decode("utf-8") == self._sig:
                num_add = h5["metadata"][1]
                A = h5["A"][:num_add]
                y = h5["y"][:num_add]
                row_type = h5["row_type"][:num_add]
                row_weight = h5["row_weight"][:num_add]
                self._acquire_datasets(num_add)
                metadata = self._h5_handle.require_dataset(
                    name="metadata", shape=(2, ), dtype=np.int32)
                row = metadata[1]
                self._ds_A[row: row + num_add] = A
                self._ds_y[row: row + num_add] = y
                self._ds_t[row: row + num_add] = row_type
                self._ds_w[row: row + num_add] = row_weight
                metadata[1] = row + num_add
                return num_add

        return 0

    @staticmethod
    def _check_weight(weight):
        if weight is None:
            return [1.0, 1.0, 1.0]
        elif np.isscalar(weight):
            return np.ones(3) * weight
        else:
            weight = np.asarray(weight).flatten()
            if len(weight) > 3:
                raise ValueError("`weight` should be a vector of size <= 3")
            return weight

    def _get_dataset(self, name, d1, dtype=np.float64):
        """
        Acquire a float64 dataset.
        """
        metadata = self._h5_handle.require_dataset(
            name="metadata", shape=(2, ), dtype=np.int32)
        size = metadata[0]
        if d1 == 0:
            shape = (size, )
            maxshape = (self._max_len, )
        else:
            shape = (size, d1)
            maxshape = (self._max_len, d1)
        ds = self._h5_handle.require_dataset(
            name=name, shape=shape, maxshape=maxshape, dtype=dtype)
        return ds

    def _acquire_datasets(self, num_add):
        """
        Acquire the coefficients' matrix `A`, the targets vector `y`, the
        weights vector `w` and the atom types vector `t`.
        """
        metadata = self._h5_handle.require_dataset(
            name="metadata", shape=(2, ), dtype=np.int32)

        size = metadata[0]
        row = metadata[1]

        self._ds_A = self._get_dataset("A", d1=self._dim)
        self._ds_y = self._get_dataset("y", d1=1)
        self._ds_w = self._get_dataset("row_weight", d1=0)
        self._ds_t = self._get_dataset("row_type", d1=0, dtype=np.int32)

        if row + num_add >= size:
            new_size = row + num_add
            if new_size >= self._max_len:
                raise IOError(
                    f"New size {new_size} exceeds max size {self._max_len}")
            metadata[0] = new_size
            self._ds_A.resize(new_size, axis=0)
            self._ds_y.resize(new_size, axis=0)
            self._ds_w.resize(new_size, axis=0)
            self._ds_t.resize(new_size, axis=0)

    def _update_one(self, weight, atoms, results, loc):
        """
        A single update.
        """
        metadata = self._h5_handle.require_dataset(
            name="metadata", shape=(2, ), dtype=np.int32)
        row = metadata[1]
        nlocal = len(atoms)
        for i, specie in enumerate(self._species):
            G_abkm = results[specie].G_abkm
            a = G_abkm.shape[0]
            if a == 0:
                continue
            ii = i * (self._bkm_val + 1)
            it = ii + (self._bkm_val + 1)
            self._ds_A[row, ii] = a
            self._ds_A[row, ii + 1: it] = G_abkm.sum(
                axis=0, keepdims=False).flatten()
            self._ds_y[row, 0] = atoms.get_potential_energy()
            self._ds_t[row] = 0
            self._ds_w[row] = weight[0]
        row += 1

        if self._use_forces:
            self._ds_A[row: row + nlocal * 3, 0] = 0.0
            for i, specie in enumerate(self._species):
                dGdrx_axbkm = results[specie].dGdrx_axbkm.reshape(
                    (nlocal * 3, self._bkm_val))
                ii = i * (self._bkm_val + 1)
                it = ii + (self._bkm_val + 1)
                self._ds_A[row: row + nlocal * 3, ii + 1: it] = dGdrx_axbkm
                forces = atoms.get_forces()
                for j in range(nlocal):
                    self._ds_y[row + loc[j] * 3 + 0, 0] = forces[j, 0]
                    self._ds_y[row + loc[j] * 3 + 1, 0] = forces[j, 1]
                    self._ds_y[row + loc[j] * 3 + 2, 0] = forces[j, 2]
                self._ds_t[row: row + nlocal * 3] = 1
                self._ds_w[row: row + nlocal * 3] = weight[1]
            row += nlocal * 3

        if self._use_stress:
            vol = atoms.get_volume()
            for i, specie in enumerate(self._species):
                dGdh_axybkm = results[specie].dGdh_axybkm
                a = dGdh_axybkm.shape[0]
                if a == 0:
                    continue
                virial = dGdh_axybkm.sum(
                    axis=0, keepdims=False).reshape((3, 3, self._bkm_val))
                stress = virial / vol
                ii = i * (self._bkm_val + 1)
                it = ii + (self._bkm_val + 1)
                self._ds_A[row + 0, ii + 1: it] = stress[0, 0]
                self._ds_A[row + 1, ii + 1: it] = stress[1, 1]
                self._ds_A[row + 2, ii + 1: it] = stress[2, 2]
                self._ds_A[row + 3, ii + 1: it] = stress[1, 2]
                self._ds_A[row + 4, ii + 1: it] = stress[0, 2]
                self._ds_A[row + 5, ii + 1: it] = stress[0, 1]
                self._ds_t[row: row + 6] = 2
                self._ds_w[row: row + 6] = weight[2]
                self._ds_y[row: row + 6, 0] = atoms.get_stress()
            row += 6

        # Write back the current row
        metadata[1] = row

    def parallel_add(self, list_of_atoms, weight=1.0, n_jobs=-1):
        """
        Add a list of `Atoms` to the database.
        """
        num_add = 0
        for atoms in list_of_atoms:
            nlocal = len(atoms)
            num_add += 1
            if self._use_forces:
                num_add += nlocal * 3
            if self._use_stress:
                num_add += 6

        weight = self._check_weight(weight)
        self._acquire_datasets(num_add)

        pipeline = partial(
            calculate_linear_mtp_parameters,
            species=self._species,
            filters=self._filters,
            rcut=self._rcut,
            max_moment=self._max_moment)

        data = joblib.Parallel(n_jobs=n_jobs, verbose=5)(
            joblib.delayed(pipeline)(atoms) for atoms in list_of_atoms)

        for idx, (results, loc, _) in enumerate(data):
            atoms = list_of_atoms[idx]
            self._update_one(weight, atoms, results, loc)

    def add(self, atoms: Atoms, weight=1.0):
        """
        Add an `Atoms` to the database.
        """
        results, loc, _ = calculate_linear_mtp_parameters(
            atoms,
            self._species,
            filters=self._filters,
            rcut=self._rcut,
            max_moment=self._max_moment)

        nlocal = len(atoms)
        num_add = 1
        if self._use_forces:
            num_add += nlocal * 3
        if self._use_stress:
            num_add += 6

        weight = self._check_weight(weight)
        self._acquire_datasets(num_add)
        self._update_one(weight, atoms, results, loc)

    def fit(self, alpha=0.0, l1_ratio=0.0, n_jobs=1, num_metrics=None,
            ridge_solver="svd"):
        """
        Linear regression with combined L1 and L2 priors as regularizer.

        Minimizes the objective function::

                1 / (2 * n_samples) * ||y - Ax||^2_2
                + alpha * l1_ratio * ||x||_1
                + 0.5 * alpha * (1 - l1_ratio) * ||x||^2_2

        If you are interested in controlling the L1 and L2 penalty
        separately, keep in mind that this is equivalent to::

                a * ||x||_1 + 0.5 * b * ||x||_2^2

        where::

                alpha = a + b and l1_ratio = a / (a + b)

        The parameter l1_ratio corresponds to alpha in the glmnet R package
        while alpha corresponds to the lambda parameter in glmnet. Specifically,
        l1_ratio = 1 is the lasso penalty. Currently, l1_ratio <= 0.01 is not
        reliable, unless you supply your own sequence of alpha.
        """
        end = self._h5_handle["metadata"][1]
        if isinstance(num_metrics, int):
            end = min(num_metrics, end)
        elif isinstance(num_metrics, float) and 0.0 < num_metrics <= 1.0:
            end = int(num_metrics * end)
        elif num_metrics is None:
            pass
        else:
            raise ValueError("`num_metrics` should be an integer or "
                             "a float within (0.0, 1.0]")
        A = self._h5_handle["A"][:end]
        y = self._h5_handle["y"][:end]
        row_weight = self._h5_handle["row_weight"][:end]
        if alpha == 0.0:
            if l1_ratio == 0.0:
                model = LinearRegression(fit_intercept=True, n_jobs=n_jobs)
            else:
                model = ElasticNet(fit_intercept=True, l1_ratio=l1_ratio,
                                   alpha=alpha)
        else:
            model = Ridge(fit_intercept=True, alpha=alpha, solver=ridge_solver)
        model.fit(A, y, sample_weight=row_weight)
        x = model.coef_.reshape((-1, 1))
        residual = np.linalg.norm(A @ x - y)
        self._h5_handle["x"][()] = x
        return x, residual

    def evaluate_mean_absolute_errors(self, x=None):
        """
        Evaluate mean absolute errors of energy (eV/atom), forces (eV/Angstrom)
        and stress (GPa).
        """
        end = self._h5_handle["metadata"][1]
        A = self._h5_handle["A"][:end]
        y = self._h5_handle["y"][:end]
        if x is None:
            x = self._h5_handle["x"]
        row_type = self._h5_handle["row_type"][:end]
        diff = np.abs(A @ x - y)
        selections = np.where(row_type == 0)[0]
        natoms = np.zeros_like(selections)
        for j, row in enumerate(selections):
            for i in range(len(self._species)):
                natoms[j] += A[row, i * (self._bkm_val + 1)]
        mae = {"energy": np.mean(diff[selections] / natoms)}
        if self._use_forces:
            mae["forces"] = np.mean(diff[np.where(row_type == 1)[0]])
        if self._use_stress:
            mae["stress"] = np.mean(diff[np.where(row_type == 2)[0]]) / GPa
        return mae

    def calculate_one(self, atoms: Atoms, x=None):
        """
        Calculate energy, forces and stress of the given `atoms`.
        """
        results, loc, reordered = calculate_linear_mtp_parameters(
            atoms,
            self._species,
            filters=self._filters,
            rcut=self._rcut,
            max_moment=self._max_moment)
        energy = 0.0
        forces = np.zeros((len(atoms), 3))
        virial = np.zeros((3, 3))
        vol = atoms.get_volume()

        if x is None:
            x = self._h5_handle["x"]

        for i, specie in enumerate(self._species):
            ii = i * (self._bkm_val + 1)
            it = ii + (self._bkm_val + 1)
            eatom = x[ii]
            c_bkm = x[ii + 1: it, 0].reshape(self._bkm_shape)
            o = results[specie]
            energy += o.G_abkm.shape[0] * eatom
            energy += np.einsum("abkm,bkm->a", o.G_abkm, c_bkm).sum()
            forces += np.einsum("axbkm,bkm->ax", o.dGdrx_axbkm, c_bkm)
            virial += np.einsum("axybkm,bkm->xy", o.dGdh_axybkm, c_bkm)

        if reordered:
            forces = forces[loc]
        stress = virial / vol
        stress = stress[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]

        return {"energy": energy, "forces": forces, "stress": stress}

    def calculate(self, atoms=None, properties=None,
                  system_changes=all_changes):
        """
        The ase.calculator.Calculator.calculate method.
        """
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)
        self.results = self.calculate_one(atoms)

    def open(self, filename):
        """
        Open a HDF5 database.
        """
        def may_convert(_val):
            if isinstance(_val, bytes):
                return _val.decode("utf-8")
            else:
                return _val

        self.close()
        filename = Path(filename)
        h5 = None
        if filename.exists():
            h5 = h5py.File(filename, mode='r+')
            sig = h5.get("sig", default=None)
            if not sig or may_convert(sig[()]) != self._sig:
                h5.close()
                from os import remove
                remove(filename)
        if h5 is None:
            self._h5_handle = h5py.File(filename, mode='a')
            self._h5_handle["sig"] = self._sig
            self._h5_handle["metadata"] = np.array(
                [10000, 0], dtype=np.int32)
            self._h5_handle["x"] = np.zeros((self._dim, 1), dtype=np.float64)
        else:
            self._h5_handle = h5
        self._database_file = filename

    def close(self):
        """
        Close the HDF5 database.
        """
        if self._h5_handle:
            self._h5_handle.close()

    def export(self, filename, precision=64):
        """
        Export the model for LAMMPS pair_style tensoralloy/native.
        """
        if precision == 64:
            dtype = np.float64
        else:
            dtype = np.float32
        masses = [atomic_masses[atomic_numbers[elt]] for elt in self._species]
        chars = []
        for elt in self._species:
            for char in elt:
                chars.append(ord(char))

        params = filter_presets[self._preset]
        data = {"rmax": dtype(self._rcut),
                "nelt": np.int32(len(self._species)),
                "masses": np.array(masses, dtype=dtype),
                "numbers": np.array(chars, dtype=np.int32),
                "tdnp": np.int32(0),
                "precision": precision,
                "use_fnn": np.int32(0),
                "descriptor::rl": np.array(params["rl"], dtype=dtype),
                "descriptor::pl": np.array(params["pl"], dtype=dtype),
                "descriptor::type": np.int32(0),
                "nlayers": np.int32(0),
                "max_moment": np.int32(self._max_moment),
                "actfn": np.int32(0),
                "fctype": np.int32(0),
                "layer_sizes": np.array([0], dtype=np.int32),
                "use_resnet_dt": np.int32(0), "apply_output_bias": np.int32(1)}

        x = self._h5_handle["x"]
        for i, elt in enumerate(self._species):
            ii = i * (self._bkm_val + 1)
            it = ii + (self._bkm_val + 1)
            eatom = x[ii]
            c_bkm = x[ii + 1: it, 0].flatten()
            data[f"weights_{i}_0"] = c_bkm
            data[f"biases_{i}_0"] = eatom

        np.savez(filename, **data)


class LinearPotentialCalculator(Calculator):
    """
    Lightweight LinearPotential calculator.
    """

    # ASE Calculator properties
    implemented_properties = ["energy", "forces", "stress"]
    nolabel = True

    def __init__(self, species: List[str], solution: np.ndarray, preset: str,
                 max_moment=3, rcut=5.0):
        """
        Initialization.
        """
        Calculator.__init__(self)
        self.species = species
        self.filters = get_filter_preset(preset)
        self.solution = np.array(solution)
        self.preset = preset
        self.max_moment = max_moment
        self.rcut = rcut
        self.bkm_shape = (len(species), len(self.filters), max_moment + 1)
        self.bkm_val = len(species) * (max_moment + 1) * len(self.filters)

    def calculate(self, atoms=None, properties=None,
                  system_changes=all_changes):
        """
        The ase.calculator.Calculator.calculate method.
        """
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)

        results, loc, reordered = calculate_linear_mtp_parameters(
            atoms,
            self.species,
            filters=self.filters,
            rcut=self.rcut,
            max_moment=self.max_moment)
        energy = 0.0
        forces = np.zeros((len(atoms), 3))
        virial = np.zeros((3, 3))
        vol = atoms.get_volume()

        for i, specie in enumerate(self.species):
            ii = i * (self.bkm_val + 1)
            it = ii + (self.bkm_val + 1)
            eatom = self.solution[ii]
            c_bkm = self.solution[ii + 1: it].reshape(self.bkm_shape)
            o = results[specie]
            energy += o.G_abkm.shape[0] * eatom
            energy += np.einsum("abkm,bkm->a", o.G_abkm, c_bkm).sum()
            forces += np.einsum("axbkm,bkm->ax", o.dGdrx_axbkm, c_bkm)
            virial += np.einsum("axybkm,bkm->xy", o.dGdh_axybkm, c_bkm)

        if reordered:
            forces = forces[loc]
        stress = virial / vol
        stress = stress[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]

        self.results["energy"] = energy
        self.results["forces"] = forces
        self.results["stress"] = stress
        
        
class TensorMDPythonCalculator(Calculator):

    # ASE Calculator properties
    implemented_properties = ["energy", "forces", "stress"]
    nolabel = True

    def __init__(self, species: List[str], solution: np.ndarray, preset: str,
                 max_moment=3, rcut=5.0):
        """
        Initialization.
        """
        Calculator.__init__(self)

        self.species = species
        self.filters = get_filter_preset(preset)
        self.solution = np.array(solution)
        self.preset = preset
        self.max_moment = max_moment
        self.rcut = rcut
        self.bkm_shape = (len(species), len(self.filters), max_moment + 1)
        self.bkm_val = len(species) * (max_moment + 1) * len(self.filters)
        self.E_a = self.solution[::self.bkm_val + 1]
        self.c_bkm = []
        for i in range(len(species)):
            i0 = i * (self.bkm_val + 1) + 1
            it = (i + 1) * (self.bkm_val + 1)
            self.c_bkm.append(self.solution[i0: it].reshape(self.bkm_shape))
    
    def _calculate(self, atoms: Atoms):
        i_l, j_l, D_lx, R_l = neighbor_list('ijDd', atoms, cutoff=self.rcut)
        T_dm = get_T_dm(self.max_moment)
        elements = atoms.get_chemical_symbols()

        # Setup dimensions
        ddim = T_dm.shape[0]
        adim = len(atoms)
        bdim = len(self.species)
        nnl = 0
        for i in range(len(atoms)):
            indices = np.where(i_l == i)[0]
            ii = atoms.numbers[i_l[indices]]
            ij = atoms.numbers[j_l[indices]]
            if len(ii) > 0:
                nnl = max(max(Counter(cantor_pairing(ii, ij)).values()), nnl)
        cdim = nnl
        kdim = len(self.filters)
        xdim = 3

        # Setup istart for each specie if `reorder` is True
        eltnum = Counter(atoms.get_chemical_symbols())
        bounds = np.zeros(len(self.species) + 1, dtype=np.int32)
        istart = Counter()
        offset = 0
        for i, specie in enumerate(self.species):
            istart[specie] = offset
            offset += eltnum[specie]
            bounds[i + 1] = offset
        i2a = np.zeros(adim, dtype=np.int32)
        a2i = np.zeros(adim, dtype=np.int32)
        for i, specie in enumerate(atoms.get_chemical_symbols()):
            i2a[i] = istart[specie]
            a2i[istart[specie]] = i
            istart[specie] += 1

        # Compute H_kl
        H_lk = np.zeros((len(i_l), kdim), dtype=np.float64)
        dH_lk = np.zeros_like(H_lk)
        s_l, ds_l = fcut(R_l, cutforce=self.rcut)
        for k in range(kdim):
            fk, dfk = self.filters[k](R_l)
            H_lk[:, k] = fk * s_l
            dH_lk[:, k] = fk * ds_l + dfk * s_l

        # Setup integer arrays: `eltypes` and `eltmap`.
        eltypes = np.zeros(adim, dtype=np.int32)
        for i in range(adim):
            eltypes[i] = self.species.index(elements[i])
        eltmap = np.zeros((bdim, bdim), dtype=np.int32)
        for i in range(bdim):
            k = 1
            for j in range(bdim):
                if i == j:
                    eltmap[i, j] = 0
                else:
                    eltmap[i, j] = k
                    k += 1

        # Initialized tensors
        R_abc = np.zeros((adim, bdim, cdim), dtype=np.float64)
        drdrx_abcx = np.zeros((adim, bdim, cdim, xdim))
        H_abck = np.zeros((adim, bdim, cdim, kdim), dtype=np.float64)
        dHdr_abck = np.zeros_like(H_abck)
        M_abcd = np.zeros((adim, bdim, cdim, ddim), dtype=np.float64)
        i_abc = np.zeros((adim, bdim, cdim), dtype=np.int32) - 1
        j_abc = np.zeros((adim, bdim, cdim), dtype=np.int32) - 1
        t_a = np.zeros(adim, dtype=np.int32)
        neigh = np.zeros((adim, bdim), dtype=np.int32)
        i_l = i_l.astype(np.int32)
        j_l = j_l.astype(np.int32)

        setup_tensors(eltmap, eltypes, i_l, j_l, neigh, i2a, R_abc, R_l, 
                      drdrx_abcx, D_lx, H_abck, H_lk, dHdr_abck, dH_lk, i_abc, 
                      j_abc, t_a, M_abcd, np.int32(self.max_moment))
        mask = (R_abc > 0).astype(np.int32)
        
        P_abkd = np.einsum('abcd,abck->abkd', M_abcd, H_abck)
        sign = np.sign(P_abkd[:, :, :, 0])
        G_abkm = np.einsum('abkd,dm->abkm', P_abkd**2, T_dm)
        G_abkm[:, :, :, 0] = np.sqrt(G_abkm[:, :, :, 0]) * sign

        energy = 0.0
        energies = np.zeros(len(atoms), dtype=np.float64)
        dEdG = np.zeros((adim, bdim, kdim, self.max_moment + 1),
                         dtype=np.float64)

        for i in range(len(self.species)):
            a0 = bounds[i]
            at = bounds[i + 1]
            if at - a0 > 0:
                y = np.einsum('abkm,bkm->a', 
                              G_abkm[a0: at], self.c_bkm[i]) + self.E_a[i]
                for a in range(a0, at):
                    energies[a2i[a]] = y[a - a0]
                    dEdG[a] = self.c_bkm[i]
        energy = energies.sum()

        dEdG[:, :, :, 0] *= 0.5 / (G_abkm[:, :, :, 0] + 1e-16)
        dEdS = np.einsum("dm,abkm,abkd->abkd", 2 * T_dm, dEdG, P_abkd)
        U_abck = np.einsum("abkd,abcd->abck", dEdS, M_abcd)
        V_abcd = np.einsum("abkd,abck->abcd", dEdS, H_abck)

        F1_abcx = np.zeros((adim, bdim, cdim, xdim), dtype=np.float64)
        F2_abcx = np.zeros((adim, bdim, cdim, xdim), dtype=np.float64)
        kernel_F1(U_abck, dHdr_abck, drdrx_abcx, mask, F1_abcx)
        kernel_F2(np.int32(self.max_moment), V_abcd, R_abc, drdrx_abcx, mask, 
                  F2_abcx)

        forces = np.zeros((len(atoms), 3), dtype=np.float64)
        virial = np.zeros((len(atoms), 6), dtype=np.float64)
        sum_forces(F1_abcx + F2_abcx, i_abc, j_abc, a2i, R_abc, drdrx_abcx, 
                   forces, virial)
        
        stresses = -virial / atoms.get_volume()
        stress = stresses.sum(axis=0)

        return energy, energies, forces, stress, stresses

    def calculate(self, atoms=None, properties=None,
                  system_changes=all_changes):
        """
        The ase.calculator.Calculator.calculate method.
        """
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)

        energy, energies, forces, stress, stresses = self._calculate(atoms)

        self.results["energy"] = energy
        self.results["forces"] = forces
        self.results["stress"] = stress
        self.results["stresses"] = stresses
        self.results["energies"] = energies


def benchmark():
    import time
    import tqdm
    from argparse import ArgumentParser
    from tensoralloy.io.db import snap
    parser = ArgumentParser()
    parser.add_argument("-m", "--max-moment",
                        default=3, type=int, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument("-r", "--rcut", default=6.0, type=float)
    parser.add_argument("-p", "--preset", default="pexp@medium")
    parser.add_argument("-n", "--num-metrics", type=float, default=1.0)
    parser.add_argument("-i", "--interval", type=int, default=1)
    args = parser.parse_args()
    db = snap()
    species = ["Mo", "Ni"]
    rcut = args.rcut
    preset = args.preset
    max_moment = args.max_moment
    hdf5_file = f"MoNi_{max_moment}_{preset}_{rcut:.1f}.hdf5"
    if os.path.exists(hdf5_file):
        os.remove(hdf5_file)
    pot = LinearPotential(
        species, preset, max_moment, rcut, database=hdf5_file)
    t0 = time.time()
    for atoms_id in tqdm.trange(1, len(db) + 1, args.interval):
        atoms = db.get_atoms(id=atoms_id, add_additional_information=True)
        pot.add(atoms)
    t1 = time.time()
    pot.fit(num_metrics=args.num_metrics)
    t2 = time.time()
    mae = pot.evaluate_mean_absolute_errors()
    print(f"Elapsed time: build = {t1 - t0:.3f}, fit = {t2 - t1:.3f}")
    print(f"MAE: energy={mae['energy'] * 1000:.2f} meV/atom, "
          f"forces={mae['forces']:.3f} eV/Ang, stress={mae['stress']:.3f} GPa")
    pot.close()
    if os.path.exists(hdf5_file):
        os.remove(hdf5_file)


def test_copy_hdf5():
    from tensoralloy.io.db import snap

    db = snap()
    kwargs = {"species": ["Mo", "Ni"], "rcut": 6.0, "max_moment": 3,
              "preset": "pexp@medium", "use_stress": True, "use_forces": True}

    h1 = LinearPotential(database="h1.hdf5", **kwargs)
    stack = []
    for i in range(1, 5):
        atoms = db.get_atoms(id=i, add_additional_information=True)
        stack.append(atoms)
    h1.parallel_add(stack, n_jobs=1)

    h2 = LinearPotential(database="h2.hdf5", **kwargs)
    stack = []
    for i in range(1, 3):
        atoms = db.get_atoms(id=i, add_additional_information=True)
        stack.append(atoms)
    h2.parallel_add(stack, n_jobs=1)
    h2.close()

    h3 = LinearPotential(database="h2a.hdf5", **kwargs)
    h3.copy_hdf5("h2.hdf5")
    stack = []
    for i in range(3, 5):
        atoms = db.get_atoms(id=i, add_additional_information=True)
        stack.append(atoms)
    h3.parallel_add(stack, n_jobs=1)

    row1 = h1._h5_handle["metadata"][1]
    row3 = h3._h5_handle["metadata"][1]

    A1 = h1._h5_handle["A"][:row1]
    A3 = h3._h5_handle["A"][:row3]

    print(np.linalg.norm(A1 - A3))


if __name__ == "__main__":
    test_copy_hdf5()
