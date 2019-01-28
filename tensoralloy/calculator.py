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
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import importer
from typing import List, Tuple
from phonopy import __version__ as phonopy_version
from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath
from phonopy.phonon.band_structure import get_band_qpoints
from phonopy.interface import get_default_physical_units
from phonopy.structure.cells import guess_primitive_matrix

from tensoralloy.transformer.base import DescriptorTransformer
from tensoralloy.transformer import SymmetryFunctionTransformer, EAMTransformer
from tensoralloy.nn.basic import exportable_properties
from tensoralloy.phonony import Phonopy

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def print_phononpy():
    """
    Print the phonopy logo.
    """
    print("""        _
  _ __ | |__   ___  _ __   ___   _ __  _   _
 | '_ \| '_ \ / _ \| '_ \ / _ \ | '_ \| | | |
 | |_) | | | | (_) | | | | (_) || |_) | |_| |
 | .__/|_| |_|\___/|_| |_|\___(_) .__/ \__, |
 |_|                            |_|    |___/""")


# noinspection PyUnresolvedReferences
def print_phonopy_version():
    """
    Print phonopy version.
    """
    version = phonopy_version
    version_text = ('%s' % version).rjust(44)
    try:
        import pkg_resources
        dist = pkg_resources.get_distribution("phonopy")
        if dist.has_version():
            ver = dist.version.split('.')
            if len(ver) > 3:
                rev = ver[3]
                version_text = ('%s-r%s' % (version, rev)).rjust(44)
    except ImportError:
        pass
    except Exception as err:
        if (err.__module__ == 'pkg_resources' and
                err.__class__.__name__ == 'DistributionNotFound'):
            pass
        else:
            raise
    finally:
        print(version_text)


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
            self._ops = self._get_ops()
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
        }
        ops = {}
        for prop, name in props_and_names.items():
            try:
                ops[prop] = graph.get_tensor_by_name(name)
            except KeyError:
                continue
        if not self._predict_properties:
            self._predict_properties = list(ops.keys())
        else:
            assert set(self._predict_properties) == set(ops.keys())
        return ops

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
            print_phononpy()
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
        Calculator.calculate(self, atoms, properties, *args)

        with self._graph.as_default():
            ops = {target: self._ops[target] for target in properties}
            self.results = self._sess.run(
                ops, feed_dict=self._transformer.get_feed_dict(atoms))


def phonon_spectrum_example():
    """
    A demo of plotting phonon spectrum of Ni/fcc.
    """
    from ase.build import bulk
    from os.path import join
    from tensoralloy.test_utils import test_dir

    band_paths = [
        [np.array([0, 0, 0]),
         np.array([1 / 2, 0, 1 / 2]),
         np.array([1 / 2, 1 / 4, 3 / 4]),
         np.array([1 / 2, 1 / 2, 1]),
         np.array([3 / 8, 3 / 8, 3 / 4]),
         np.array([0, 0, 0]),
         np.array([1 / 2, 1 / 2, 1 / 2])]
    ]

    labels = [
        r"$\Gamma$",
        r"$\mathrm{X}$",
        r"$\mathrm{W}$",
        r"$\mathrm{X}$",
        r"$\mathrm{K}$",
        r"$\Gamma$",
        r"$\mathrm{L}$",
    ]

    atoms = bulk('Ni')
    calc = TensorAlloyCalculator(join(test_dir(), 'Ni', 'Ni.zjw04xc.pb'))
    atoms.calc = calc
    calc.get_phonon_spectrum(
        atoms,
        primitive_axes='auto',
        supercell=(4, 4, 4),
        band_paths=band_paths,
        band_labels=labels,
        npoints=101,
        use_wavenumber=True, plot_vertical_lines=True,
        verbose=True)


if __name__ == "__main__":
    phonon_spectrum_example()
