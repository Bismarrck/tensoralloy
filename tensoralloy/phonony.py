# coding=utf-8
"""
This module defines a custom implementation of `phonopy.Phonopy`.
"""
from __future__ import print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np

from phonopy import Phonopy as base_Phonopy
from phonopy.phonon import band_structure
from phonopy import __version__ as phonopy_version
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.ticker import MultipleLocator
from typing import List

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


class BandStructure(band_structure.BandStructure):
    """
    A customized `BandStructure` class.
    """

    def plot(self, ax: ImageGrid, labels=None, path_connections=None,
             is_legacy=False, use_wavenumber=False, plot_vertical_lines=False):
        """
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
