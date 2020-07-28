# coding=utf-8
"""
This module defines a custom implementation of `phonopy.Phonopy`.
"""
from __future__ import print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import warnings
import glob

from ase import Atoms
from ase.io import read
from os.path import join, exists
from os import remove

try:

    from phonopy import Phonopy as base_Phonopy
    from phonopy.phonon import band_structure
    from phonopy import __version__ as phonopy_version

    from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath
    from phonopy.phonon.band_structure import get_band_qpoints
    from phonopy.structure.atoms import PhonopyAtoms
    from phonopy.interface import parse_disp_yaml
    from phonopy.interface import get_default_physical_units
    from phonopy.interface import get_default_displacement_distance
    from phonopy.interface import write_FORCE_SETS
    from phonopy.interface.vasp import write_vasp
    from phonopy.structure.cells import guess_primitive_matrix
    from phonopy.units import VaspToCm
    from phonopy import file_IO

except ImportError:
    raise ImportError(
        "To use `PhononCalculator` phonopy-1.14.2 must be installed")

from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.ticker import MultipleLocator
from typing import List

from tensoralloy.calculator import TensorAlloyCalculator

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def print_phonopy():
    """
    Print the phonopy logo.
    """
    print(r"""        _
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


class BandStructure(band_structure.BandStructure):
    """
    A customized `BandStructure` class.
    """

    def plot(self, ax: ImageGrid, labels=None, path_connections=None,
             is_legacy=False, use_wavenumber=False, plot_vertical_lines=False):
        r"""
        The plot function.

        Parameters
        ----------
        ax : ImageGrid
            The `ImageGrid` instance for plotting this band.
        labels : None or List[str]
            A list of str as the labels of the high-symmetry positions.
            For example `['$\Gamma$', '$\mathrm{X}$']`.
        path_connections : None or List[bool]
            A list of bool indicating whether two adjacent points should be
            connected or not.
        is_legacy : bool
            If True, use the legacy `plot` function.
        use_wavenumber : bool
            If True, frequencies will be converted to cm-1. Defauls to False so
            that THz will be used. The legacy plot function does not support
            this.
        plot_vertical_lines : bool
            If True, vertical lines at each high-symmetry point will be plotted.
            The legacy plot function does not support this.

        """
        if is_legacy:
            self._plot_legacy(ax, labels=labels)
        else:
            self._plot(ax, labels=labels, path_connections=path_connections,
                       use_wavenumber=use_wavenumber,
                       plot_vertical_lines=plot_vertical_lines)

    def _plot(self, axs: ImageGrid, labels=None, path_connections=None,
              use_wavenumber=False, plot_vertical_lines=False):
        if path_connections is None:
            connections = [True, ] * len(self._paths)
        else:
            connections = path_connections

        if not use_wavenumber:
            frequencies = self._frequencies
        else:
            factor = 33.35641
            frequencies = [x * factor for x in self._frequencies]

        max_freq = np.max(frequencies)
        max_dist = np.max(self._distances)
        scale = max_freq / max_dist * 1.5
        distances = [d * scale for d in self._distances]

        # T T T F F -> [[0, 3], [4, 4]]
        lefts = [0]
        rights = []
        for i, c in enumerate(connections):
            if not c:
                lefts.append(i + 1)
                rights.append(i)
        seg_indices = [list(range(l, r + 1)) for l, r in zip(lefts, rights)]
        special_points = []
        for indices in seg_indices:
            pts = [distances[i][0] for i in indices]
            pts.append(distances[indices[-1]][-1])
            special_points.append(pts)

        if use_wavenumber:
            axs[0].set_ylabel(r'Frequency ($\mathrm{cm}^{-1}$)', fontsize=14)
        else:
            axs[0].set_ylabel(r'Frequency (THz)', fontsize=14)

        l_count = 0
        for ax, spts in zip(axs, special_points):
            ax.xaxis.set_ticks_position('both')
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_tick_params(which='both', direction='in')
            ax.yaxis.set_tick_params(which='both', direction='in')
            ax.set_xlim(spts[0], spts[-1])
            ax.set_xticks(spts)
            if labels is None:
                ax.set_xticklabels(['', ] * len(spts),
                                   fontdict={'fontsize': 13})
            else:
                ax.set_xticklabels(labels[l_count:(l_count + len(spts))],
                                   fontdict={'fontsize': 13})
                l_count += len(spts)
            ax.plot([spts[0], spts[-1]], [0, 0],
                    linestyle=':', linewidth=0.5, color='b')
            if use_wavenumber:
                ax.yaxis.set_major_locator(MultipleLocator(50.0))

        count = 0
        for d, f, c in zip(distances, frequencies, connections):
            ax = axs[count]
            ax.plot(d, f, 'r-', linewidth=1)
            if not c:
                count += 1

        if plot_vertical_lines:
            count = 0
            for d, c in zip(distances, connections):
                if d[0] == 0:
                    continue
                ax = axs[count]
                ymax = ax.get_ylim()[1]
                ax.axvline(d[0], ymin=0, ymax=ymax, linestyle='--',
                           color='black', linewidth=0.75)
                if not c:
                    count += 1


class Phonopy(base_Phonopy):
    """
    A customized `Phonopy` class.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializtion method.
        """
        super(Phonopy, self).__init__(*args, **kwargs)
        self._band_structure = None

    def set_band_structure(self,
                           bands,
                           is_eigenvectors=False,
                           is_band_connection=False):
        """
        Initialize a `BandStructure` object.

        Parameters
        ----------
        bands : List of ndarray
            Sets of qpoints that can be passed to phonopy.set_band_structure().
            shape of each ndarray : (npoints, 3)
        is_eigenvectors : bool
            Flag whether eigenvectors are calculated or not.
        is_band_connection : bool
            Flag whether each band is connected or not. This is achieved by
            comparing similarity of eigenvectors of neghboring poins. Sometimes
            this fails.

        """
        if self._dynamical_matrix is None:
            print("Warning: Dynamical matrix has not yet built.")
            self._band_structure = None
            return False

        self._band_structure = BandStructure(
            bands,
            self._dynamical_matrix,
            is_eigenvectors=is_eigenvectors,
            is_band_connection=is_band_connection,
            group_velocity=self._group_velocity,
            factor=self._factor)
        return True

    def plot_band_structure(self,
                            labels=None,
                            path_connections=None,
                            is_legacy=True,
                            use_wavenumber=False,
                            plot_vertical_lines=False):
        """
        Plot the band.
        """
        if labels:
            from matplotlib import rc
            rc('text', usetex=True)

        if is_legacy:
            fig, axs = plt.subplots(1, 1)
        else:
            n = len([x for x in path_connections if not x])
            fig = plt.figure()
            axs = ImageGrid(fig, 111,  # similar to subplot(111)
                            nrows_ncols=(1, n),
                            axes_pad=0.1,
                            add_all=True,
                            label_mode="L")
        self._band_structure.plot(axs,
                                  labels=labels,
                                  path_connections=path_connections,
                                  is_legacy=is_legacy,
                                  use_wavenumber=use_wavenumber,
                                  plot_vertical_lines=plot_vertical_lines)
        fig.tight_layout()

        return plt


class PhononCalculator(TensorAlloyCalculator):
    """
    A special `TensorAlloyCalculator` which focuses on phonon functions.
    """

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

    def _parse_set_of_forces(self, forces_filenames):
        """
        A helper function to compute atomic forces of displaced structures.
        """
        force_sets = []
        for filename in forces_filenames:
            atoms = read(filename, format='vasp', index=-1)
            force_sets.append(self.get_forces(atoms))
        return force_sets

    @staticmethod
    def _write_supercells_with_displacements(supercell,
                                             cells_with_displacements,
                                             pre_filename="POSCAR",
                                             sposcar_filename="SPOSCAR",
                                             width=3):
        """
        A wrapper of the original function.
        """
        write_vasp(sposcar_filename, supercell, direct=True)
        for i, cell in enumerate(cells_with_displacements):
            if cell is not None:
                write_vasp("{pre_filename}-{0:0{width}}".format(
                    i + 1,
                    pre_filename=pre_filename,
                    width=width),
                    cell,
                    direct=True)

    def get_numeric_force_sets(self, phonon: Phonopy, supercell: Atoms,
                               displacement_distance=None,
                               force_sets_zero_mode=False,
                               verbose=True):
        """
        Compute the force constants using the numeric method.
        """
        force_sets_filename = join(self._model_dir, "FORCE_SETS")
        disp_filename = join(self._model_dir, "disp.yaml")
        poscar_prefilename = join(self._model_dir, "POSCAR")
        sposcar_filename = join(self._model_dir, "SPOSCAR")

        if displacement_distance is None:
            displacement_distance = get_default_displacement_distance('vasp')

        phonon.generate_displacements(
            distance=displacement_distance,
            is_plusminus=False,
            is_diagonal=False,
            is_trigonal=False)
        displacements = phonon.get_displacements()

        _supercell = PhonopyAtoms(supercell.symbols,
                                  positions=supercell.positions,
                                  pbc=supercell.pbc,
                                  cell=supercell.cell)

        file_IO.write_disp_yaml(displacements, _supercell, disp_filename)

        # Write supercells with displacements
        cells_with_disps = phonon.get_supercells_with_displacements()
        self._write_supercells_with_displacements(
            supercell=_supercell,
            cells_with_displacements=cells_with_disps,
            pre_filename=poscar_prefilename,
            sposcar_filename=sposcar_filename,
        )

        disp_dataset = parse_disp_yaml(filename=disp_filename)
        num_displacements = len(disp_dataset['first_atoms'])
        n_atoms = disp_dataset['natom']
        force_filenames = glob.glob(join(self._model_dir, "POSCAR-*"))

        if force_sets_zero_mode:
            num_displacements += 1
        force_sets = self._parse_set_of_forces(force_filenames)

        if force_sets:
            if force_sets_zero_mode:
                def _subtract_residual_forces(_force_sets):
                    for i in range(1, len(_force_sets)):
                        _force_sets[i] -= _force_sets[0]
                    return _force_sets[1:]
                force_sets = _subtract_residual_forces(force_sets)
            for forces, disp in zip(force_sets, disp_dataset['first_atoms']):
                disp['forces'] = forces
            write_FORCE_SETS(disp_dataset, filename=force_sets_filename)

        if verbose:
            if force_sets:
                print("%s has been created." % force_sets_filename)
            else:
                print("%s could not be created." % force_sets_filename)

        phonon.set_displacement_dataset(
            file_IO.parse_FORCE_SETS(n_atoms, filename=force_sets_filename))

        # Need to calculate full force constant tensors
        phonon.produce_force_constants(use_alm=False)

        # Remove files
        filenames = force_filenames
        filenames += [disp_filename, force_sets_filename, sposcar_filename]
        for afile in filenames:
            if exists(afile):
                remove(afile)

    def get_phonon_spectrum(self, atoms=None, supercell=(4, 4, 4),
                            numeric_hessian=False, primitive_axes=None,
                            band_paths=None, band_labels=None, npoints=51,
                            symprec=1e-5, save_fc=None, image_file=None,
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
        numeric_hessian : bool
            If False, the analytical hessian will be computed. Otherwise the
            numeric approch shall be used.
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
        symprec : float
            The precision for determining the spacegroup of the primitive.
        save_fc : None or str
            The file to save the calculated force constants or None.
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

        # Physical units: energy,  distance,  atomic mass, force
        # vasp          : eV,      Angstrom,  AMU,         eV/Angstrom
        # tensoralloy   : eV,      Angstrom,  AMU,         eV/Angstrom
        physical_units = get_default_physical_units('vasp')

        # Check the primitive matrix
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

        calc = self.__class__(self._graph_model_path)
        super_atoms = atoms * supercell
        super_atoms.calc = calc
        calc.calculate(super_atoms)

        if not numeric_hessian:
            hessian = calc.get_property('hessian', super_atoms)
            clf = calc.transformer.get_vap_transformer(super_atoms)
            fc = clf.reverse_map_hessian(hessian, phonopy_format=True)

            # Convert the matrix to `double` and save it if required.
            fc = fc.astype(np.float64)
            if save_fc is not None:
                np.savez(save_fc, fc=fc)
            phonon.set_force_constants(fc)

        else:
            self.get_numeric_force_sets(phonon, super_atoms, None)

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
