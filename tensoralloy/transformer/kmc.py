#!coding=utf-8
"""
The transformer for TensorKMC
"""
import tensorflow as tf
import numpy as np

from ase import Atoms
from ase.data import chemical_symbols
from ase.neighborlist import neighbor_list
from collections import Counter
from typing import List, Dict
from ase.build import bulk

from tensoralloy.utils import get_kbody_terms
from tensoralloy.transformer.base import DescriptorTransformer
from tensoralloy.precision import get_float_dtype
from tensoralloy import atoms_utils

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class KMCTransformer(DescriptorTransformer):
    """
    The universal transformer for all models.
    """

    def __init__(self, elements: List[str], rcut, nnl_max: int):
        """
        The initialization method

        Parameters
        ----------
        elements : List[str]
            A list of str as the ordered unique elements.
        rcut : float
            The cutoff radius for radial interactions.
        nnl_max : int
            `nnl_max` should be a constant for KMC.

        """
        DescriptorTransformer.__init__(self)

        for element in elements:
            if element not in chemical_symbols:
                raise ValueError(f"{element} is not a valid chemical symbol!")

        all_kbody_terms, kbody_terms_for_element, elements = \
            get_kbody_terms(elements, angular=False, symmetric=False)

        radial_kbody_terms = []
        for kbody_term in all_kbody_terms:
            radial_kbody_terms.append(kbody_term)

        max_nr_terms = 0
        for element in elements:
            max_nr_terms = max(
                max_nr_terms, len(kbody_terms_for_element[element]))

        radial_interactions = {}
        for element in elements:
            kbody_terms = kbody_terms_for_element[element]
            for i, kbody_term in enumerate(kbody_terms):
                radial_interactions[kbody_term] = i

        self._all_kbody_terms = all_kbody_terms
        self._kbody_terms_for_element = kbody_terms_for_element
        self._max_nr_terms = max_nr_terms
        self._radial_interactions = radial_interactions
        self._rcut = rcut
        self._elements = elements
        self._nnl_max = nnl_max
        self._n_elements = len(elements)

    def as_dict(self) -> Dict:
        """
        Return a JSON serializable dict representation of this transformer.
        """
        return {'class': self.__class__.__name__, 'elements': self._elements,
                'rcut': self._rcut, 'nnl_max': self._nnl_max}

    @property
    def descriptor(self):
        """
        Return the descriptor name. This property will be removed soon.
        """
        return "kmc"

    @property
    def rc(self):
        """
        Return the cutoff radius. This property will be removed soon.
        """
        return self._rcut

    @property
    def rcut(self) -> float:
        """
        Return the cutoff radius for radial interactions.
        """
        return self._rcut

    @property
    def elements(self) -> List[str]:
        """
        Return a list of str as the ordered unique elements.
        """
        return self._elements

    @property
    def n_elements(self) -> int:
        """
        Return the total number of unique elements.
        """
        return self._n_elements

    @property
    def max_occurs(self):
        """
        There is no restriction for the occurances of an element.
        """
        return {el: np.inf for el in self._elements}

    @property
    def use_computed_dists(self):
        """
        Return True if interatomic distances shall be computed or given
        directly.
        """
        return True

    @property
    def all_kbody_terms(self):
        """
        A list of str as the ordered k-body terms.
        """
        return self._all_kbody_terms

    @property
    def kbody_terms_for_element(self) -> Dict[str, List[str]]:
        """
        A dict of (element, kbody_terms) as the k-body terms for each type of
        elements.
        """
        return self._kbody_terms_for_element

    def _check_keys(self, features: dict):
        """
        Make sure `placeholders` contains enough keys.
        """
        assert 'etemperature' in features

        for element in self._elements:
            assert f"G/{element}" in features
            assert f"G/{element}/masks" in features
            assert f"atom_masks/{element}" in features
            assert f"n_atoms_vap/{element}" in features

    def build_radial_graph(self, features: dict):
        """
        Build the computation graph for calculating radial descriptors.
        """
        with tf.name_scope("Radial"):
            descriptors = {}
            for element in self._elements:
                descriptors[element] = (features[f"G/{element}"],
                                        features[f"G/{element}/masks"])
            return descriptors

    def _get_atom_masks(self, features: dict):
        # `atom_masks` indicates whether the corresponding atom is real or
        # virtual.
        atom_masks_dict = {}
        for element in self._elements:
            atom_masks_dict[element] = features[f"atom_masks/{element}"]
        return atom_masks_dict

    def build_graph(self, features: dict):
        """
        Build the graph for computing universal descriptors.

        Returns
        -------
        ops : Dict[str, Tuple[tf.Tensor, tf.Tensor]]
            A dict of {element: (descriptor, mask)}.
        """
        self._check_keys(features)
        with tf.name_scope(f"Transformer"):
            g2 = self.build_radial_graph(features)
            atom_masks = self._get_atom_masks(features)
            return {"radial": g2, "angular": None, "atom_masks": atom_masks}

    def _create_float_5d(self, dtype, name):
        return self._get_or_create_placeholder(
            dtype,
            name=name,
            shape=(4, self._max_nr_terms, None, self._nnl_max, 1))

    def _create_float_4d(self, dtype, name):
        return self._get_or_create_placeholder(
            dtype,
            name=name,
            shape=(self._max_nr_terms, None, self._nnl_max, 1))

    def _initialize_placeholders(self):
        """
        Initialize placeholder tensors.
        """
        with tf.name_scope("Placeholders/"):
            dtype = get_float_dtype()

            self._placeholders["atom_masks"] = self._create_float_1d(
                dtype=dtype, name='atom_masks')
            self._placeholders["etemperature"] = self._create_float(
                dtype=dtype, name='etemperature')

            for element in self._elements:
                self._placeholders[f"G/{element}"] = self._create_float_5d(
                    dtype, f"G/{element}")
                self._placeholders[f"G/{element}/masks"] = \
                    self._create_float_4d(dtype, f"G/{element}/masks")
                self._placeholders[f"atom_masks/{element}"] = \
                    self._create_float_1d(name=f"atom_masks/{element}",
                                          dtype=dtype)
                self._placeholders[f"n_atoms_vap/{element}"] = \
                    self._create_float(dtype=dtype,
                                       name=f"n_atoms_vap/{element}")

        return self._placeholders

    def get_np_feed_dict(self, atoms: Atoms):
        """
        Return a dict of features (Numpy or Python objects).
        """
        dtype = get_float_dtype().as_numpy_dtype
        itype = np.int32
        symbols = atoms.get_chemical_symbols()
        vap = self.get_vap_transformer(atoms)
        atom_masks = vap.atom_masks.astype(dtype)
        etemp = atoms_utils.get_electron_temperature(atoms)

        offsets = {}
        offset = 1
        for element in self._elements:
            offsets[element] = offset
            offset += vap.max_occurs[element]

        g = np.zeros((4,
                      self._max_nr_terms,
                      vap.max_vap_natoms,
                      self._nnl_max,
                      1), dtype=dtype)
        m = np.zeros((self._max_nr_terms,
                      vap.max_vap_natoms,
                      self._nnl_max,
                      1), dtype=dtype)

        ilist, jlist, rij, dij = neighbor_list('ijdD', atoms, self._rcut)
        counters = {}
        for index in range(len(ilist)):
            atomi = ilist[index]
            atomj = jlist[index]
            if atomi not in counters:
                counters[atomi] = Counter()
            symboli = symbols[atomi]
            symbolj = symbols[atomj]
            d1 = self._radial_interactions[f"{symboli}{symbolj}"]
            d2 = vap.local_to_gsl_map[atomi + 1]
            d3 = counters[atomi][d1]
            g[0, d1, d2, d3, 0] = rij[index]
            g[1, d1, d2, d3, 0] = dij[index, 0]
            g[2, d1, d2, d3, 0] = dij[index, 1]
            g[3, d1, d2, d3, 0] = dij[index, 2]
            m[d1, d2, d3, 0] = 1.0
            counters[atomi][d1] += 1

        feed_dict = dict(atom_masks=atom_masks, etemperature=dtype(etemp))
        for element in self._elements:
            n = vap.max_occurs.get(element, 0)
            if n == 0:
                virt4d = [self._max_nr_terms, 1, self._nnl_max, 1]
                virt5d = [4, ] + virt4d
                feed_dict[f"atom_masks/{element}"] = np.zeros(1, dtype=itype)
                feed_dict[f"n_atoms_vap/{element}"] = itype(1)
                feed_dict[f"G/{element}"] = np.zeros(virt5d, dtype=dtype)
                feed_dict[f"G/{element}/masks"] = np.zeros(virt4d, dtype=dtype)
            else:
                d20 = offsets[element]
                d2t = offsets[element] + n
                feed_dict[f"atom_masks/{element}"] = np.ones(n, dtype=itype)
                feed_dict[f"n_atoms_vap/{element}"] = itype(n)
                feed_dict[f"G/{element}"] = g[:, :, d20: d2t, :, :]
                feed_dict[f"G/{element}/masks"] = m[:, d20: d2t, :, :]
        return feed_dict

    def get_feed_dict(self, atoms: Atoms):
        """
        Return the feed dict.
        """
        feed_dict = {}

        if not self._placeholders:
            self._initialize_placeholders()
        placeholders = self._placeholders

        for key, value in self.get_np_feed_dict(atoms).items():
            feed_dict[placeholders[key]] = value

        return feed_dict

    def get_constant_features(self, atoms: Atoms):
        """
        Return a dict of constant feature tensors for the given `Atoms`.
        """
        feed_dict = dict()
        with tf.name_scope("Constants"):
            for key, val in self.get_np_feed_dict(atoms).items():
                feed_dict[key] = tf.convert_to_tensor(val, name=key)
        return feed_dict


def test():
    from tensorflow_estimator import estimator as tf_estimator
    from tensoralloy.transformer.universal import UniversalTransformer
    from tensoralloy.neighbor import find_neighbor_size_of_atoms
    from tensoralloy.nn import EamAlloyNN

    atoms = bulk('Cu', crystalstructure='fcc', a=2.855, cubic=True)
    atoms.set_chemical_symbols(['Cu', 'Fe', 'Cu', 'Cu'])
    atoms *= [2, 2, 2]
    atoms.positions += np.random.randn(len(atoms), 3) * 0.1

    elements = ['Cu', 'Fe']
    rc = 6.5

    nn = EamAlloyNN(elements,
                    custom_potentials='zjw04',
                    export_properties=['energy'],
                    minimize_properties=['energy'])

    with tf.Graph().as_default():
        utf = UniversalTransformer(elements, rcut=rc)
        nn.attach_transformer(utf)
        op = nn.build(utf.get_constant_features(atoms),
                      tf_estimator.ModeKeys.PREDICT)['energy']
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            e1 = sess.run(op)

    with tf.Graph().as_default():
        nnl = find_neighbor_size_of_atoms(atoms, rc=rc).nnl
        ktf = KMCTransformer(elements, rcut=rc, nnl_max=nnl)
        nn.attach_transformer(ktf)
        op = nn.build(ktf.get_constant_features(atoms),
                      tf_estimator.ModeKeys.PREDICT)['energy']
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            e2 = sess.run(op)

    assert abs(e1 - e2) < 1e-8


if __name__ == "__main__":
    test()
