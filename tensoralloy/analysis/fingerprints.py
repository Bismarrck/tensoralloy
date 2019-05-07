# coding=utf-8
"""
This script implements Oganov's Fingerprints and the similarity kernel.

The code is (probably) written by Lasse B. Vilhelmsen.
"""

from __future__ import print_function

import numpy as np

from itertools import product, combinations_with_replacement
from math import erf, sqrt
from copy import copy
from typing import Union, List


class FingerprintsComparator:
    """
    Implementation of structural similarity comparison based on Oganov's
    fingerprint function:

    * http://dx.doi.org/10.1063/1.3079326
    * http://dx.doi.org/10.1016/j.cpc.2010.06.007

    The advantages of cosine distances are that (1) they can only take values
    between 0 and 1, enabling universal distance criteria (e.g. 0.01 – very
    small distance, 0.20 – large, 0.5 – very large distance) and (2) cosine
    distances are less affected by the “distance concentration effects” that
    generally create trouble in multidimensional spaces.

    """

    def __init__(self, atomic_numbers: Union[List[int], np.ndarray], n_top=None,
                 cell=None, dE=1.0, cos_dist_max=5e-3, rcut=20., binwidth=0.05,
                 pbc=(True, True, True), maxdims=(0, 0, 0), sigma=0.025,
                 nsigma=4):
        """
        Initialization method.

        Parameters
        ----------
        atomic_numbers : array_like
            The ordered atomic numbers of ths system.
        n_top : int
            The number of atoms to optimize (everything except the substrate).
        dE : float
            Energy difference above which two structures are automatically
            considered to be different.
        cell : np.ndarray
            The base cell matrix to use.
        cos_dist_max : float
            The maximal cosine distance between two structures in order to
            be still considered the same structure.
        rcut : float
            The cut-off radius for the fingerprints.
        binwidth : float
            Width of the bins over which the fingerprints are discretized.
        pbc: List[bool]
            Conditions (PBC) along each of the three unit cell vectors
            when calculating the fingerprint.

            Note: for isolated systems (pbc = [False,False,False]), the pair
            correlation function itself is always short-ranged (i.e. decays to
            zero beyond a certain radius), so unity is not substracted for
            calculating the fingerprint. Also the volume normalization
            disappears.
        maxdims : List[int] or array_like
            If PBC in only 1 or 2 dimensions are specified, the maximal
            thicknesses along the non-periodic directions must be specified, as
            a list of length 3 (the values for the periodic directions are not
            read).

            Note: in this implementation, the cell vectors of the non-
            periodic directions are assumed to be orthogonal amongst
            themselves and also orthogonal to the cell vectors of the
            periodic directions.
        sigma : float
            Standard deviation of the gaussian smearing to be applied in the
            calculation of the fingerprints (in Angstrom).
        nsigma : int
            The distance (as the number of standard deviations sigma) at which
            the gaussian smearing is cut off (i.e. no smearing beyond that
            distance).

        """

        if cell is None:
            raise ValueError('You need to provide the simulation cell as well!')

        self.n_top = n_top or 0
        self.cell = cell
        self.dE = dE
        self.cos_dist_max = cos_dist_max
        self.rcut = rcut
        self.binwidth = binwidth
        self.pbc = pbc
        self.maxdims = maxdims
        self.sigma = sigma
        self.nsigma = nsigma

        self.dimensions = self.pbc.count(True)
        if self.dimensions == 1 or self.dimensions == 2:
            for direction in range(3):
                if not self.pbc[direction]:
                    if not self.maxdims[direction] > 0:
                        e = '''A positive thickness must be specified in 
                               maxdims for all non-periodic directions when 
                               applying PBC to only one or two directions.'''
                        raise ValueError(e)

        # determining the volume normalization factor
        pbc_dirs = [i for i in range(3) if self.pbc[i]]
        non_pbc_dirs = [i for i in range(3) if not self.pbc[i]]

        if self.dimensions == 3:
            self.volume = abs(np.dot(np.cross(self.cell[0, :],
                                              self.cell[1, :]), self.cell[2, :]))

        elif self.dimensions == 2:
            non_pbc_dir = non_pbc_dirs[0]

            a = np.cross(self.cell[pbc_dirs[0], :], self.cell[pbc_dirs[1], :])
            b = self.maxdims[non_pbc_dir]
            b /= np.linalg.norm(self.cell[non_pbc_dir, :])

            self.volume = np.abs(np.dot(a, b * self.cell[non_pbc_dir, :]))

        elif self.dimensions == 1:
            pbc_dir = pbc_dirs[0]

            v0 = self.cell[non_pbc_dirs[0], :]
            b0 = self.maxdims[non_pbc_dirs[0]]
            b0 /= np.linalg.norm(self.cell[non_pbc_dirs[0], :])
            v1 = self.cell[non_pbc_dirs[1], :]
            b1 = self.maxdims[non_pbc_dirs[1]]
            b1 /= np.linalg.norm(self.cell[non_pbc_dirs[1], :])

            self.volume = np.abs(np.dot(np.cross(b0 * v0, b1 * v1),
                                        self.cell[pbc_dir, :]))

        elif self.dimensions == 0:
            self.volume = 1.

        # Find the unique atomic numbers
        unique_types = sorted(list(set(atomic_numbers)))
        self.typedic = {}
        for t in unique_types:
            tlist = [i for i, ni in enumerate(atomic_numbers) if ni == t]
            self.typedic[t] = tlist

    def looks_like(self, a1, a2):
        """
        Return if structure a1 or a2 are similar or not.
        """
        if len(a1) != len(a2):
            raise Exception('The two configurations are not the same size.')

        # first we check the energy criteria
        if a1.get_calculator() is not None and a2.get_calculator() is not None:
            dE = abs(a1.get_potential_energy() - a2.get_potential_energy())
            if dE >= self.dE:
                return False

        # then we check the structure
        cos_dist = self._compare_structure_(a1, a2)

        if cos_dist < self.cos_dist_max:
            return True
        else:
            return False

    def __json_encode__(self, fingerprints):
        """
        json does not accept tuples as dict keys, so in order to write the
        fingerprints to atoms.info, we need to convert the (A,B) tuples to
        strings
        """
        fingerprints_encoded = {}
        for key, val in fingerprints.items():
            newkey = "_".join(map(str, list(key)))
            fingerprints_encoded[newkey] = val
        self.typedic_encoded = {}
        for key, val in self.typedic.items():
            newkey = str(key)
            self.typedic_encoded[newkey] = val
        return [fingerprints_encoded, self.typedic_encoded]

    def __json_decode__(self, fingerprints):
        """
        This is the reverse operation of __json_encode__
        """
        fingerprints_decoded = {}
        for key, val in fingerprints.items():
            newkey = tuple(map(int, key.split("_")))
            fingerprints_decoded[newkey] = np.array(val)
        self.typedic_decoded = {}
        for key, val in self.typedic.items():
            newkey = int(key)
            self.typedic_decoded[newkey] = val
        return [fingerprints_decoded, self.typedic_decoded]

    def _compare_structure_(self, a1, a2):
        """ Returns the cosine distance between the two structures,
            using their fingerprints. """

        if len(a1) != len(a2):
            raise Exception('The two configurations are not the same size.')

        a1top = a1[-self.n_top:]
        a2top = a2[-self.n_top:]

        if 'fingerprints' in a1.info:
            fp1, self.typedic = a1.info['fingerprints']
            fp1, self.typedic = self.__json_decode__(fp1)
        else:
            fp1 = self.get_features(a1top)
            a1.info['fingerprints'] = self.__json_encode__(fp1)

        if 'fingerprints' in a2.info:
            fp2, self.typedic = a2.info['fingerprints']
            fp2, self.typedic = self.__json_decode__(fp2)
        else:
            fp2 = self.get_features(a2top)
            a2.info['fingerprints'] = self.__json_encode__(fp2)

        if sorted(fp1) != sorted(fp2):
            raise AssertionError('The two structures have fingerprints \
                                  with different compounds.')

        cos_dist = self.get_similarity(fp1, fp2)

        return cos_dist

    def get_features(self, atoms, individual=False):
        """
        Returns a [fingerprints,self.typedic] list, where fingerprints is a
        dictionary with the fingerprints and self.typedic is a dictionary with
        the list of atom indices for each element (or "type") in the atoms
        object. The keys in the fingerprints dictionary are the (A,B) tuples,
        which are the different element-element combinations in the atoms object
        (A and B are the atomic numbers). When A != B, the (A,B) tuple is sorted
        (A < B). As an example, an object with atoms of elements X,Y and Z
        will have (X,X), (X,Y), (X,Z), (Y,Y), (Y,Z) and (Z,Z) fingerprints.

        If individual=True, a dict is returned, where each atom index has an
        {atomic_number: fingerprint} dict as value. So, if individual=False,
        these fingerprints from atoms of the same atomic number are summed
        together."""

        pos = atoms.get_positions()
        scalpos = atoms.get_scaled_positions()
        num = atoms.get_atomic_numbers()
        cell = atoms.cell

        unique_types = sorted(list(set(num)))  # the unique atomic numbers
        posdic = {}
        for t in unique_types:
            tlist = [i for i, atom in enumerate(atoms) if atom.number == t]
            posdic[t] = pos[tlist]

        # Setting up the required extra parameters if we don't want to apply
        # PBC in 1 or 2 directions:
        non_pbc_dirs = [i for i in range(3) if not self.pbc[i]]

        if self.dimensions == 2:
            non_pbc_dir = non_pbc_dirs[0]

            b = self.maxdims[non_pbc_dir]
            b /= np.linalg.norm(cell[non_pbc_dir, :])

            maxpos = np.max(scalpos[:, non_pbc_dir])
            minpos = np.min(scalpos[:, non_pbc_dir])
            pwidth = maxpos - minpos
            pmargin = 0.5 * (b - pwidth)
            # note: here is a place where we assume that the non-periodic
            # direction is orthogonal to the periodic ones:
            pmin = np.min(scalpos[:, non_pbc_dir]) - pmargin
            pmin *= np.linalg.norm(cell[non_pbc_dir, :])
            pmax = np.max(scalpos[:, non_pbc_dir]) + pmargin
            pmax *= np.linalg.norm(cell[non_pbc_dir, :])

        elif self.dimensions == 1:
            # pbc_dir = pbc_dirs[0]

            b0 = self.maxdims[non_pbc_dirs[0]]
            b0 /= np.linalg.norm(cell[non_pbc_dirs[0], :])
            b1 = self.maxdims[non_pbc_dirs[1]]
            b1 /= np.linalg.norm(cell[non_pbc_dirs[1], :])

            # note: here is a place where we assume that the non-periodic
            # direction is orthogonal to the periodic ones:
            maxpos = np.max(scalpos[:, non_pbc_dirs[0]])
            minpos = np.min(scalpos[:, non_pbc_dirs[0]])
            pwidth = maxpos - minpos
            pmargin = 0.5 * (b0 - pwidth)

            pmin = np.min(scalpos[:, non_pbc_dirs[0]]) - pmargin
            pmin *= np.linalg.norm(cell[non_pbc_dirs[0], :])
            pmax = np.max(scalpos[:, non_pbc_dirs[0]]) + pmargin
            pmax *= np.linalg.norm(cell[non_pbc_dirs[0], :])

            # maxpos = np.max(scalpos[:, non_pbc_dirs[1]])
            # minpos = np.min(scalpos[:, non_pbc_dirs[1]])
            # qwidth = maxpos - minpos
            qmargin = 0.5 * (b1 - pwidth)

            qmin = np.min(scalpos[:, non_pbc_dirs[1]]) - qmargin
            qmin *= np.linalg.norm(cell[non_pbc_dirs[1], :])
            qmax = np.max(scalpos[:, non_pbc_dirs[1]]) + qmargin
            qmax *= np.linalg.norm(cell[non_pbc_dirs[1], :])

        def arccos(x):
            """
            A specific arccos function. The domain of the numpy version is
            only [-1, 1].
            """
            return (1. / 1j) * np.log(x + sqrt(x ** 2 - 1))

        def surface_area_0d(r):
            """
            Return the surface area of the 0D lattice.
            """
            return 4 * np.pi * (r ** 2)

        def surface_area_1d(r, _pos):
            """
            Return the surface area of the 1D lattice.
            """
            q0 = _pos[non_pbc_dirs[1]]
            phi1 = arccos((qmax - q0) / r).real
            phi2 = np.pi - arccos((qmin - q0) / r).real
            factor = 1 - (phi1 + phi2) / np.pi
            return surface_area_2d(r, _pos) * factor

        def surface_area_2d(r, _pos):
            """
            Return the surface area of the 2D lattice.
            """
            p0 = _pos[non_pbc_dirs[0]]
            return 2 * np.pi * r * (min(pmax - p0, r) + min(p0 - pmin, r))

        def surface_area_3d(r):
            """
            Return the surface area of the 3D lattice.
            """
            return 4 * np.pi * (r ** 2)

        # determining which neighbouring cells to 'visit' for a given rcut:
        cell_vec_norms = np.apply_along_axis(np.linalg.norm, 0, cell)
        neighbours = []
        for i in range(3):
            ncellmax = int(np.ceil(abs(self.rcut / cell_vec_norms[i]))) + 1
            if self.pbc[i]:
                neighbours.append(range(-ncellmax, ncellmax + 1))
            else:
                neighbours.append([0])
        # neighbours = eg [[-2,-1,0,1,2],[-1,0,1],[0]]

        # parameters for the binning:
        m = int(np.ceil(self.nsigma * self.sigma / self.binwidth))
        smearing_norm = erf(
            0.25 * np.sqrt(2) * self.binwidth * (2 * m + 1) * 1. / self.sigma)
        nbins = int(np.ceil(self.rcut * 1. / self.binwidth))

        def take_individual_rdf(index, unique_type):
            """
            Compute the radial distribution function of atoms of type
            `unique_type` around the atom with index `index`.
            """
            rdf = np.zeros(nbins)
            for nx, ny, nz in product(*neighbours):
                displacement = np.dot(cell.T, np.array([nx, ny, nz]).T)
                displaced_pos = posdic[unique_type] + displacement
                deltaRs = np.apply_along_axis(np.linalg.norm, 1,
                                              displaced_pos - pos[index])

                if min(deltaRs) > self.rcut + self.nsigma * self.sigma:
                    # for neighbouring cells this far away from the original
                    # cell, we will find no neighbours less than rcut away.
                    continue

                for deltaR in deltaRs:
                    rbin = int(np.floor(deltaR / self.binwidth))
                    for i in range(-m, m + 1):
                        newbin = rbin + i
                        if newbin < 0 or newbin >= nbins:
                            continue

                        c = 0.25 * np.sqrt(2) * self.binwidth * 1. / self.sigma
                        va = 0.5 * erf(c * (2 * i + 1))
                        vb = 0.5 * erf(c * (2 * i - 1))
                        # dividing by smearing_norm ensures that the sum of the
                        # values over the different bins we have smeared out,
                        # equals 1
                        value = (va - vb) / smearing_norm

                        if deltaR < 1e-6:
                            # We're looking at the central atom. Discard.
                            value = 0
                        else:
                            if self.dimensions == 3:
                                area = surface_area_3d(deltaR)
                                value /= area * self.binwidth
                            elif self.dimensions == 2:
                                area = surface_area_2d(deltaR, pos[index])
                                value /= area * self.binwidth
                            elif self.dimensions == 1:
                                area = surface_area_1d(deltaR, pos[index])
                                value /= area * self.binwidth
                            elif self.dimensions == 0:
                                area = surface_area_0d(deltaR)
                                value /= area * self.binwidth

                        rdf[newbin] += value

            rdf /= len(self.typedic[unique_type]) * 1. / self.volume
            return rdf

        if individual:
            fingerprints = []
            for i in range(len(atoms)):
                u1 = atoms[i].number
                fingerprints.append({})
                for u2 in unique_types:
                    fingerprint = take_individual_rdf(i, u2)
                    if self.dimensions > 0:
                        fingerprint -= 1
                    fingerprints[i][tuple(sorted((u1, u2)))] = fingerprint
                type_combinations = combinations_with_replacement(
                    unique_types, r=2)
                for type1, type2 in type_combinations:
                    key = (type1, type2)
                    if key not in fingerprints[i]:
                        fingerprints[i][key] = np.zeros(nbins) - 1

        else:
            fingerprints = {}
            type_combinations = combinations_with_replacement(unique_types, r=2)
            for type1, type2 in type_combinations:
                key = (type1, type2)
                fingerprint = np.zeros(nbins)
                for i in self.typedic[type1]:
                    fingerprint += take_individual_rdf(i, type2)
                fingerprint /= len(self.typedic[type1])
                if self.dimensions > 0:
                    fingerprint -= 1
                fingerprints[key] = fingerprint

        return fingerprints

    def get_similarity(self, fp1, fp2):
        """
        Returns the cosine distance from two fingerprints. It also needs
        information about the number of atoms from each element.
        """
        keys = sorted(fp1)

        # calculating the weights:
        w = {}
        wtot = 0
        for key in keys:
            weight = len(self.typedic[key[0]]) * len(self.typedic[key[1]])
            wtot += weight
            w[key] = weight
        for key in keys:
            w[key] *= 1. / wtot

        # calculating the fingerprint norms:
        norm1 = 0
        norm2 = 0
        for key in keys:
            norm1 += (np.linalg.norm(fp1[key]) ** 2) * w[key]
            norm2 += (np.linalg.norm(fp2[key]) ** 2) * w[key]
        norm1 = np.sqrt(norm1)
        norm2 = np.sqrt(norm2)

        # calculating the distance:
        distance = 0
        for key in keys:
            distance += np.sum(fp1[key] * fp2[key]) * w[key] / (norm1 * norm2)

        distance = 0.5 * (1 - distance)

        return distance

    def get_motifs(self, atoms):
        """
        Find and return the motifs.
        """
        pos = atoms.get_positions()
        motifs = []
        for i in range(len(atoms)):
            motif = copy(atoms)
            posi = pos[i]
            for j in range(len(atoms)):
                if i == j:
                    continue
                posj = pos[j]
                dist = np.linalg.norm(posi - posj)
                if dist > self.rcut:
                    del motif[j]
            motifs.append(motif)

        return motifs


def usage_demo():
    """
    A simple demo of `FingerprintsComparator`.
    """
    from ase.io import read
    from os.path import join
    from tensoralloy.test_utils import test_dir

    a, b = read(join(test_dir(), 'B28.xyz'), format='xyz', index='0:2')
    a.cell = np.eye(3) * 20.0
    b.cell = np.eye(3) * 20.0

    comparator = FingerprintsComparator(
        a.numbers,
        n_top=len(a),
        cell=a.cell,
        rcut=6.0,
        binwidth=0.02,
        pbc=[False] * 3)

    fa = comparator.get_features(a)
    fb = comparator.get_features(b)
    print(comparator.get_similarity(fa, fb))


if __name__ == "__main__":
    usage_demo()
