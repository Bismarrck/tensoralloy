#!/usr/bin/env python3
import re
from dataclasses import dataclass
from subprocess import Popen, PIPE
from pathlib import Path


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
        elapsed = self.get_vasp_elapsed_time(self.outcar)
        if elapsed < 0:
            return None
        mpi, omp = self.get_vasp_mpi_omp_ranks(self.outcar)
        device = self.get_vasp_running_device(self.outcar)
        if device == "gpu":
            hours = mpi * elapsed / 3600.0
        else:
            hours = mpi * omp * elapsed / 3600.0
        return ServiceUnit(device, mpi, elapsed, hours)
    
    def check_vasp_job_scf_convergence(self) -> bool:
        """
        Check the SCF convergence of a VASP job.
        """
        outcar = Path(outcar)
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
                if ln > 2000:
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
