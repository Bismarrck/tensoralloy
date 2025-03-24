#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# VASPKit: A Python module for VASP helper functions.
import numpy as np
import re
from dataclasses import dataclass
from subprocess import Popen, PIPE
from pathlib import Path

# TODO: 1. move `read_vasp_xml` and `atoms_utils` to tensordb package
# TODO: 2. make Vaspjob a subclass of ase.calculators.vasp.Vasp
from tensoralloy.io.vasp import read_vasp_xml


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


@dataclass(frozen=True)
class ServiceUnit:
    """
    The data structure for estimating VASP costs.

    Attributes
    ----------
    device : str
        The device type, cpu or gpu.
    n : int
        The number of CPUs or GPUs used for this task.
    t : float
        The walltime in seconds for this task.
    hours : float
        The total deivce hours.

    """

    device: str = "cpu"
    n: int = 0
    elapsed: float = 0.0
    hours: float = 0.0


class VaspJob:

    def __init__(self, jobdir: Path) -> None:
        self.jobdir = jobdir
        self.outcar = jobdir / "OUTCAR"
        self.vaspxml = jobdir / "vasprun.xml"
        self.params = {}
    
    def get_incar_parameter(self, key: str) -> str:
        """
        Return the string value of a given INCAR parameter.
        """
        if not self.params:
            self._parse_incar()
        return self.params.get(key.lower(), None)        

    def get_atoms(self, index=-1, is_finite_temperature=False):
        """
        Get the atoms object of a VASP job.
        """
        if not self.vaspxml.exists():
            return None
        return next(read_vasp_xml(self.vaspxml, index=index, 
                                  finite_temperature=is_finite_temperature))

    def _parse_incar(self):
        """
        A private method to parse the INCAR file of a VASP job.
        """
        incar = self.jobdir / "INCAR"
        if not incar.exists():
            return
        with open(incar) as fp:
            for ln, line in enumerate(fp):
                if ln == 0:
                    continue
                line = line.strip()
                if line.startswith("#"):
                    continue
                if line == "":
                    continue
                key, value = line.split("=")
                self.params[key.strip().lower()] = value.strip()

    def get_running_device(self) -> str:
        """
        Return the deivce (cpu or gpu) used to run the VASP job.
        """
        cmd = f"head -n 10 {str(self.outcar)} | grep GPUs"
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        p.communicate()
        if p.returncode == 0:
            return "gpu"
        else:
            return "cpu"
    
    def get_vasp_mpi_omp_ranks(self):
        """
        Get the number of mpi and openmp ranks of a VASP job given its OUTCAR.
        """
        cmd = f"head -n 10 {str(self.outcar)} | grep mpi-ranks"
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        stdout, _ = p.communicate()
        if p.returncode == 0:
            values = stdout.decode().split()
            mpi = int(values[1])
            omp = int(values[4])
            return mpi, omp
        else:
            return 0, 0
    
    def get_vasp_elapsed_time(self):
        """
        Get the elapsed time (seconds) of a VASP job given its OUTCAR.
        """
        cmd = f"tail -n 10 {str(self.outcar)} | grep Elapsed"
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        stdout, _ = p.communicate()
        if p.returncode != 0:
            return -1.0
        return float(stdout.decode().split()[3])
            
    def get_vasp_job_service_unit(self) -> ServiceUnit:
        """
        Get the service unit (SU) of a completed VASP job.
        """
        if not self.outcar.exists():
            return None
        elapsed = self.get_vasp_elapsed_time()
        if elapsed < 0:
            return None
        mpi, omp = self.get_vasp_mpi_omp_ranks()
        device = self.get_running_device()
        if device == "gpu":
            hours = mpi * elapsed / 3600.0
        else:
            hours = mpi * omp * elapsed / 3600.0
        return ServiceUnit(device, mpi, elapsed, hours)
    
    def check_vasp_job_scf_convergence(self) -> bool:
        """
        Check the SCF convergence of a VASP job.
        """
        outcar = Path(self.outcar)
        oszicar = outcar.parent / "OSZICAR"
        if not oszicar.exists():
            return False
        ediff_patt = re.compile(
            r"\s+EDIFF\s+=\s+([0-9\-+E.]+)\s+stopping-criterion.*")
        scf_patt = re.compile(
            r"(DAV|RMM):\s+(\d+)\s+([0-9\-+E.]+)\s+([0-9\-+E.]+)\s+(.*)")
        ediff = 1e-6
        with open(outcar) as fp:
            for ln, line in enumerate(fp):
                m = ediff_patt.search(line)
                if m:
                    ediff = float(m.group(1))
                    break
                if ln > 5000:
                    return False
        dE = ediff * 10.0
        with open(oszicar) as fp:
            for line in fp:
                m = scf_patt.search(line)
                if m:
                    dE = abs(float(m.group(4)))
        if dE <= ediff:
            return True
        else:
            return False
    
    def get_band_occupation(self):
        """
        Get the band occupation of a VASP job.
        """
        outcar = Path(self.outcar)
        if not outcar.exists():
            return None
        nkpts_patt = re.compile(r"k-points\s+NKPTS\s=\s+(\d+).*NBANDS=\s+(\d+)")
        nsw_patt = re.compile(r"\s+NSW\s+=\s+(\d+)\s+number\sof\ssteps\sfor\sIOM")
        headers_patt = re.compile(r"\sband\sNo.\s+band\senergies\s+occupation")
        ispin_patt = re.compile(r"\s+ISPIN\s+=\s+(\d)\s+spin\spolarized\scalculation")
        nbands = None
        nkpoints = None
        occupations = None
        ikpt = 0
        ispin = -1
        iband = 0
        stage = 0
        with open(outcar) as fp:
            for line in fp:
                if stage == 0:
                    m = nkpts_patt.search(line)
                    if m:
                        nkpoints = int(m.group(1))
                        nbands = int(m.group(2))
                        occupations = np.zeros((nkpoints, nbands))
                        stage = 1
                elif stage == 1:
                    m = ispin_patt.search(line)
                    if m:
                        ispin = int(m.group(1))
                        stage = 2
                elif stage == 2:
                    m = nsw_patt.search(line)
                    if m: 
                        if int(m.group(1)) > 1:
                            raise ValueError("The current implementation of "
                                             "get_band_occupation() only works for "
                                             "single point calculation!")
                        stage = 3
                elif stage == 3:
                    m = headers_patt.search(line)
                    if m:
                        ikpt += 1
                        stage = 4
                elif stage == 4:
                    values = line.split()
                    if len(values) != 3:
                        raise ValueError("Failed to parse the band occupation!")
                    iband = int(values[0])
                    occupations[ikpt - 1, iband - 1] = float(values[2])
                    if iband == nbands:
                        if ikpt == nkpoints:
                            break
                        stage = 3
        return ispin, occupations
    
    def check_band_occ(self, threshold=0.0001, check_lowest=False):
        """
        Check if the band occupations are reasonable.
        The occupation number of the highest band should be close to zero.
        For high-temperature calculations, the occupation number of the lowest band 
        should be close to one (spin localized) or two (non spin localized).
        """
        ispin, occupations = self.get_band_occupation()
        if occupations[:, -1].max() > threshold:
            return False
        if check_lowest:
            if occupations[:, 0].min() < ispin * 0.99:
                return False
        return True
