# coding=utf-8
"""
This module defines various neural network based calculators.
"""
from __future__ import print_function, absolute_import

import tensorflow as tf
import numpy as np
import json
import warnings

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.units import GPa
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer
from typing import List, Tuple
from collections import namedtuple
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.elasticity.elastic import ElasticTensor

from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath
from phonopy.phonon.band_structure import get_band_qpoints
from phonopy.interface import get_default_physical_units
from phonopy.structure.cells import guess_primitive_matrix
from phonopy.units import VaspToCm

from tensoralloy.tests.test_phonon import phonon_spectrum_example
from tensoralloy.transformer.base import DescriptorTransformer
from tensoralloy.transformer import SymmetryFunctionTransformer, EAMTransformer
from tensoralloy.nn.basic import exportable_properties
from tensoralloy.analysis.phonopy import Phonopy
from tensoralloy.analysis.phonopy import print_phonopy_version, print_phonopy
from tensoralloy.dtypes import set_float_precision, get_float_precision

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


# A collection of elastic properties.
ElasticProperty = namedtuple("ElasticProperty",
                             ("elastic_tensor", "compliance_tensor",
                              "bulk_voigt", "bulk_reuss", "bulk_vrh",
                              "shear_voigt", "shear_reuss", "shear_vrh",
                              "universal_anisotropy", "poisson_ratio"))


class TensorAlloyCalculator(Calculator):
    """
    ASE-Calculator for TensorAlloy derived protobuf models.
    """

    implemented_properties = [prop.name for prop in exportable_properties]
    default_parameters = {}
    nolabel = True

    def __init__(self, graph_model_path: str, label=None, atoms=None):
        """
        Initialization method.
        """
        super(TensorAlloyCalculator, self).__init__(
            restart=None, ignore_bad_restart_file=False, label=label,
            atoms=atoms)

        graph = tf.Graph()

        with graph.as_default():

            output_graph_def = graph_pb2.GraphDef()
            with open(graph_model_path, "rb") as fp:
                output_graph_def.ParseFromString(fp.read())
                importer.import_graph_def(output_graph_def, name="")

            self._graph_model_path = graph_model_path
            self._sess = tf.Session(
                config=tf.ConfigProto(allow_soft_placement=True),
                graph=graph)
            self._graph = graph
            self._transformer = self._get_transformer()
            self._ops, self._fp_precision = self._get_ops()
            self.implemented_properties = self._predict_properties

    @property
    def elements(self) -> List[str]:
        """
        Return a list of str as the supported elements.
        """
        return self._transformer.elements

    @property
    def transformer(self) -> DescriptorTransformer:
        """
        Return the `DescriptorTransformer` of this calculator.
        """
        return self._transformer

    @property
    def predict_properties(self):
        """
        Return a list of str as the predictable properties.
        """
        return self._predict_properties

    def get_model_timestamp(self):
        """
        Return the timestamp when the graph model was generated or None.
        """
        try:
            op = self._graph.get_tensor_by_name('Metadata/timestamp:0')
        except Exception:
            return None
        return self._sess.run(op).decode('utf-8')

    def _get_transformer(self):
        """
        Recover a `DescriptorTransformer` from the graph.
        """
        params = json.loads(self._sess.run(
            self._graph.get_tensor_by_name('Transformer/params:0')))
        if 'class' in params:
            cls = params.pop('class')
        else:
            cls = 'SymmetryFunctionTransformer'
        if 'predict_properties' in params:
            self._predict_properties = params.pop('predict_properties')
        else:
            self._predict_properties = []
        if cls == 'SymmetryFunctionTransformer':
            return SymmetryFunctionTransformer(**params)
        elif cls == 'EAMTransformer':
            return EAMTransformer(**params)
        else:
            raise ValueError(f"Unknown transformer: {cls}")

    def _get_ops(self):
        """
        Return a dict of output Ops.
        """
        graph = self._graph
        props_and_names = {
            'energy': 'Output/Energy/energy:0',
            'forces': 'Output/Forces/forces:0',
            'stress': 'Output/Stress/Voigt/stress:0',
            'total_pressure': 'Output/Pressure/pressure:0',
            'hessian': 'Output/Hessian/hessian:0',
            'elastic': 'Output/Elastic/Cijkl/elastic:0'
        }
        ops = {}
        for prop, name in props_and_names.items():
            try:
                ops[prop] = graph.get_tensor_by_name(name)
            except KeyError:
                continue
        self._predict_properties = list(ops.keys())
        if ops['energy'].dtype == tf.float32:
            fp_precision = 'medium'
        else:
            fp_precision = 'high'
        return ops, fp_precision

    def get_magnetic_moment(self, atoms=None):
        """
        This calculator cannot predict magnetic moments.
        """
        return None

    def get_magnetic_moments(self, atoms=None):
        """
        This calculator cannot predict magnetic moments.
        """
        return None

    def get_hessian(self, atoms=None):
        """
        Return the Hessian matrix.

        Returns
        -------
        hessian : array_like
            The second-order derivatives of E w.r.t R, the Hessian matrix.
            The shape is [3 * N, 3 * N].

        """
        hessian = self.get_property('hessian', atoms)
        clf = self.transformer.get_index_transformer(atoms)
        return clf.reverse_map_hessian(hessian)

    def get_forces(self, atoms=None):
        """
        Return the atomic forces.
        """
        forces = np.insert(self.get_property('forces', atoms), 0, 0, 0)
        clf = self.transformer.get_index_transformer(atoms)
        return clf.map_forces(forces, reverse=True)

    def get_stress(self, atoms=None, voigt=True):
        """
        Return the stress tensor.

        Parameters
        ----------
        atoms : Atoms
            The target `Atoms` object.
        voigt : bool
            If True, return the stress tensor in Voigt order. Otherwise the 3x3
            matrix will be returned.

        Returns
        -------
        stress : array_like
            The stress tensor in ASE internal unit, 'eV/Angstrom**3'.

        """
        if atoms is None:
            atoms = self.atoms
        stress = self.get_property('stress', atoms)
        if not voigt:
            xx, yy, zz, yz, xz, xy = stress
            stress = np.array([[xx, xy, xz],
                               [xy, yy, yz],
                               [xz, yz, zz]])
        return stress
    
    def get_total_pressure(self, atoms=None):
        """
        Return the external pressure of the target `Atoms`.

        Parameters
        ----------
        atoms : Atoms
            The target `Atoms`.

        Returns
        -------
        total_pressure : float
            The total pressure, in GPa.

        """
        stress = self.get_stress(atoms)
        return np.mean(stress[:3]) * (-1.0) / GPa

    def get_elastic_constant_tensor(self,
                                    atoms=None,
                                    auto_conventional_standard=False):
        """
        Return the elastic constant tensor C.

        Parameters
        ----------
        atoms : Atoms
            The target `Atoms`.
        auto_conventional_standard : bool
            If True, `atoms` will be converted to its corresponding conventional
            standard structure if possible.

        Returns
        -------
        elastic_tensor : ElasticTensor
            The elastic constant tensor.

        References
        ----------
        https://wiki.materialsproject.org/Elasticity_calculations

        """
        atoms = atoms or self.atoms
        assert atoms.pbc.all()

        if auto_conventional_standard:
            structure = AseAtomsAdaptor.get_structure(atoms)
            analyzer = SpacegroupAnalyzer(structure)
            atoms = AseAtomsAdaptor.get_atoms(
                analyzer.get_conventional_standard_structure())

        elastic = self.get_property('elastic', atoms, allow_calculation=True)
        return ElasticTensor.from_voigt(elastic)

    def get_frequencies_and_normal_modes(self, atoms=None):
        """
        Return the vibrational frequencies and normal modes.

        If `atoms` is periodic, this method just calculates the vibrational
        frequencies at the gamma point in the original unit cell.

        Parameters
        ----------
        atoms : Atoms
            The target `Atoms` object.

        Returns
        -------
        frequencies : array_like
            The vibrational frequencies in ascending order. The unit is cm-1.
        normal_modes : array_like
            The corresponding modes. Each row, `normal_modes[i]`, represents a
            vibration mode.

        """
        hessian = self.get_hessian(atoms)
        if atoms is None:
            atoms = self.atoms
        masses = atoms.get_masses()
        inverse_matrix = np.zeros(len(atoms) * 3)
        for i in range(len(atoms)):
            inverse_matrix[i * 3: i * 3 + 3] = masses**(-0.5)
        inverse_matrix = np.diag(inverse_matrix)
        mass_weighted_hessian = inverse_matrix @ hessian @ inverse_matrix
        eigvals, eigvecs = np.linalg.eigh(mass_weighted_hessian)
        wave_numbers = eigvals * VaspToCm
        return wave_numbers, eigvecs.transpose()

    def get_phonon_spectrum(self, atoms=None, supercell=(4, 4, 4),
                            primitive_axes=None, band_paths=None,
                            band_labels=None, npoints=51, image_file=None,
                            use_wavenumber=False, plot_vertical_lines=False,
                            verbose=False):
        """
        Plot the phonon spectrum of the target system.

        Parameters
        ----------
        atoms : Atoms
            The target `Atoms` object.
        supercell : tuple or list or array_like
            The supercell expanding factors.
        primitive_axes : None or str or array_like
            The primitive axes. If None or 'auto', the primitive axes will be
            guessed.
        band_paths: str or List[array_like]
            Sets of end points of paths or a str.

            If `band_paths` is a `str`, it must be 'auto' and band paths will be
            generated by `seekpath` automatically.

            If `band_paths` is a `List`, it should be a 3D list of shape
            `(sets of paths, paths, 3)`.

            example:
                [[[0, 0, 0], [0.5, 0.5, 0], [0.5, 0.5, 0.5]],
                [[0.5, 0.25, 0.75], [0, 0, 0]]]
        band_labels : List[str]
            Labels of the end points of band paths.
        npoints: int
            Number of q-points in each path including end points.
        image_file : str or None
            A filepath for saving the phonon spectrum if provided.
        use_wavenumber : bool
            If True, frequencies will be converted to cm-1. Defauls to False so
            that THz will be used. The legacy plot function does not support
            this.
        plot_vertical_lines : bool
            If True, vertical lines at each high-symmetry point will be plotted.
            The legacy plot function does not support this.
        verbose : bool
            A boolean. If True, more intermediate details will be logged.

        """
        if not all(atoms.pbc):
            raise ValueError("The PBC of `atoms` are all False.")
        if 'hessian' not in self._predict_properties:
            raise ValueError(
                f"{self._graph_model_path} cannot predict hessians.")

        calc = self.__class__(self._graph_model_path)
        super_atoms = atoms * supercell
        super_atoms.calc = calc
        calc.calculate(super_atoms)

        hessian = calc.get_property('hessian', super_atoms)
        clf = calc.transformer.get_index_transformer(super_atoms)
        fc = clf.reverse_map_hessian(hessian, phonopy_format=True)

        # Physical units: energy,  distance,  atomic mass, force
        # vasp          : eV,      Angstrom,  AMU,         eV/Angstrom
        # tensoralloy   : eV,      Angstrom,  AMU,         eV/Angstrom
        physical_units = get_default_physical_units('vasp')

        # Check the primitive matrix
        symprec = 1e-5
        if primitive_axes is None or primitive_axes == 'auto':
            primitive_matrix = guess_primitive_matrix(atoms, symprec)
            is_primitive_axes_auto = True
        else:
            primitive_matrix = primitive_axes
            auto_primitive_matrix = guess_primitive_matrix(atoms, symprec)
            if np.abs(primitive_matrix - auto_primitive_matrix).max() > 1e-6:
                warnings.warn("The primitive matrix differs from guessed "
                              "primitive matrix significantly",
                              category=UserWarning)
            is_primitive_axes_auto = False

        supercell_matrix = (np.eye(3) * supercell).astype(int)
        phonon = Phonopy(
            atoms,
            supercell_matrix=supercell_matrix,
            primitive_matrix=primitive_matrix,
            factor=physical_units['factor'],
            frequency_scale_factor=None,
            dynamical_matrix_decimals=None,
            force_constants_decimals=None,
            symprec=symprec,
            is_symmetry=True,
            use_lapack_solver=False,
            log_level=0)
        phonon.set_force_constants(fc)

        is_band_mode_auto = False

        if band_paths is None or band_paths == 'auto':
            if verbose:
                print("SeeK-path is used to generate band paths.")
                print("About SeeK-path https://seekpath.readthedocs.io/ "
                      "(citation there-in)")
            bands, labels, path_connections = get_band_qpoints_by_seekpath(
                phonon.primitive.to_tuple(), npoints)
            is_band_mode_auto = True
        else:
            bands = get_band_qpoints(band_paths, npoints)
            path_connections = []
            for paths in band_paths:
                path_connections += [True, ] * (len(paths) - 2)
                path_connections.append(False)
            labels = band_labels

        if verbose:
            print_phonopy()
            print_phonopy_version()
            if is_band_mode_auto:
                print("Band structure mode (Auto)")
            else:
                print("Band structure mode")
            print("Settings:")
            print("  Supercell: %s" % np.diag(supercell_matrix))
            if is_primitive_axes_auto:
                print("  Primitive matrix (Auto):")
            else:
                print("  Primitive matrix:")
            for v in primitive_matrix:
                print("    %s" % v)
            print("Spacegroup: %s" %
                  phonon.get_symmetry().get_international_table())
            print("Reciprocal space paths in reduced coordinates:")
            for band in bands:
                # noinspection PyStringFormat
                print("[%5.2f %5.2f %5.2f] --> [%5.2f %5.2f %5.2f]" %
                      (tuple(band[0]) + tuple(band[-1])))

        phonon.set_band_structure(
            bands, is_eigenvectors=False, is_band_connection=False)

        plot = phonon.plot_band_structure(
            labels=labels,
            path_connections=path_connections,
            is_legacy=False,
            use_wavenumber=use_wavenumber,
            plot_vertical_lines=plot_vertical_lines)
        if image_file is not None:
            plot.savefig(image_file)
        else:
            plot.show()

    def calculate(self, atoms=None, properties=('energy', 'forces'), *args):
        """
        Calculate the total energy and other properties (1body, kbody, atomic).

        Parameters
        ----------
        atoms : `Atoms`
            An `ase.Atoms` object to calculate.
        properties : Tuple[str]
            A list of str as the properties to calculate. Available options
            are: 'energy', 'atomic', '1body' and 'kbody'.

        """
        original = get_float_precision()

        Calculator.calculate(self, atoms, properties, *args)
        with self._graph.as_default():
            set_float_precision(self._fp_precision)
            ops = {target: self._ops[target] for target in properties}
            self.results = self._sess.run(
                ops, feed_dict=self._transformer.get_feed_dict(atoms))
            set_float_precision(original)


if __name__ == "__main__":
    phonon_spectrum_example()
