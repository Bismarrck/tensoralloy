#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from subprocess import Popen, PIPE
from ase.units import kB
from ase.calculators.vasp import Vasp
from pathlib import Path
from tensordb.utils import getitem, asarray_or_eval, scalar2array
from tensordb.vaspkit import VaspJob


class VaspAimdSampler:
    """
    The fundamental ab initio molecular dynamics simulation (AIMD) sampler.
    """

    def __init__(self, config: dict):
        self.config = config
        self.species = self.config["species"]
        self.phases = self.config["phases"]

    def init_vasp(self):
        """
        Initialize the base VASP calculator.
        """
        params = getitem(self.config, ["vasp", "pot"])

        # Setup the POTCARs path
        os.environ["VASP_PP_PATH"] = params["pp_path"]

        # Setup PORCAR for each specie.
        # For ASE VASP calculator, only the suffix is used.
        setups = {}
        if "potcars" in params:
            for i, potcar in enumerate(params["potcars"]):
                if potcar != self.species[i]:
                    setups[self.species[i]] = potcar[len(self.species[i]):]
        else:
            setups["base"] = "recommended"
        xc = params.get("xc", "pbe")

        # The NBANDS for VASP calculations.
        self.vasp_nbands = {}

        # Initialize the VASP calculator for AIMD sampling jobs.
        params = getitem(self.config, ["vasp", "sampling"])
        self.vasp = Vasp(
            xc=xc,
            setups=setups,
            ediff=params.get("ediff", 1e-5),
            lreal=params.get("lreal", "Auto"),
            prec=params.get("prec", "Normal"),
            encut=params.get("encut", 400),
            ismear=params.get("ismear", 1),
            sigma=params.get("sigma", 0.1),
            algo=params.get("algo", "normal"),
            isym=params.get("isym", 0),
            nelmin=params.get("nelmin", 4),
            isif=params.get("isif", 2),
            ibrion=params.get("ibrion", 0),
            nsw=params.get("nsw", 5000),
            potim=params.get("potim", 1),
            nwrite=params.get("nwrite", 1),
            lcharg=params.get("lcharg", False),
            lwave=params.get("lwave", False),
            nblock=params.get("nblock", 1),
            maxmix=params.get("maxmix", 60),
        )
        if "npar" in params:
            self.vasp.set(npar=params["npar"])
        if "kpar" in params:
            self.vasp.set(kpar=params["kpar"])
        if "ncore" in params:
            self.vasp.set(ncore=params["ncore"])
        self.vasp_nbands["sampling"] = params.get("nbands", None)

    # --------------------------------------------------------------------------
    # Iterators
    # --------------------------------------------------------------------------

    def sampling_task_iterator(self):
        """
        Iterate through all AIMD sampling job dirs.
        """
        workdir = self.root / "sampling"
        return workdir.glob("*/n[pv]t/*/*_*K_to_*K")

    def accurate_dft_calc_iterator(self):
        """
        Iterate through all high precision dft calculation job dirs.
        """
        workdir = self.root / "calc"
        return workdir.glob("*atoms/group*/task*")

    # --------------------------------------------------------------------------
    # Generate VASP AIMD sampling jobs
    # --------------------------------------------------------------------------

    def init_vasp_sampling_parameters(self, npt=False):
        """
        Setup the sampling parameters for VASP.
        """
        params = getitem(self.config, ["vasp", "sampling"])
        gamma = params.get("langevin_gamma", 10)

        # MDALGO = 3 for Langevin thermostat
        # ISIF = 2 for NVT ensemble
        # LANGEVIN_GAMMA: Langevin friction coefficient for atom species
        self.vasp.set(mdalgo=3, langevin_gamma=[gamma] * len(self.species))

        # Special parameters for Parrinello-Rahman NPT.
        # ISIF = 3 for NPT ensemble.
        # LANGEVIN_GAMMA_L: Langevin friction coefficient for lattice degrees of
        # freedom.
        # PMASS (optional tag for VASP) is ignored.
        if npt:
            npt_params = getitem(params, ["npt"])
            gamma = npt_params.get("langevin_gamma_l", 10)
            self.vasp.set(isif=3, langevin_gamma_l=gamma)

    @staticmethod
    def _get_nvt_sampling_task(
        jobdir: Path, t0: float, t1: float, natoms: int, override=False
    ):
        """
        Returns (id, name) for the NVT sampling task.
        If the returned id is -1, the task should be skipped.
        """
        files = [x.name for x in jobdir.glob(f"*_{natoms}atoms_*K_to_*K")]
        key = f"{natoms}atoms_{t0:.0f}K_to_{t1:.0f}K"
        exists = False
        for afile in files:
            if afile.endswith(key):
                exists = True
                break
        if exists and not override:
            return -1, None
        if override:
            newid = len(files)
        else:
            newid = len(files) + 1
        return newid, f"{newid}_{key}"

    @staticmethod
    def _get_temperatures(args: dict, vt_method: str, size: int, npt=False):
        """
        Check the temperature related arguments.
        """
        if "temperatures" in args:
            if "tstart" in args or "tstop" in args:
                raise ValueError("Cannot specify both temperatures and tstart/tstop")
            tarray = asarray_or_eval(args["temperatures"])
            if vt_method == "grid":
                return tarray, tarray
            else:
                if len(tarray) != size:
                    raise ValueError("The length of temperatures should be "
                                     "equal to volumes")
                return tarray, tarray
        else:
            if "tstart" not in args or "tstop" not in args:
                raise ValueError(
                    "Either temperatures or tstart/tstop should be specified")
            if vt_method == "grid":
                raise ValueError("Cannot specify tstart/tstop for grid VT method")
            if isinstance(args["tstart"], str):
                t0 = asarray_or_eval(args["tstart"])
            else:
                t0 = scalar2array(args["tstart"], size, "tstart", "volumes")
            if isinstance(args["tstop"], str):
                t1 = asarray_or_eval(args["tstop"])
            else:
                t1 = scalar2array(args["tstop"], size, "tstop", "volumes")
            if len(t0) != size or len(t1) != size:
                raise ValueError("The length of tstart/tstop should be equal to "
                                 "volumes")
            return t0, t1

    def create_vasp_sampling_nvt_tasks(self, override=False):
        """
        Create VASP Langeven NVT sampling jobs: gamma-only.
        """
        workdir = self.root / "sampling"
        workdir.mkdir(exist_ok=True)
        batch_jobs = []

        # The NVT vasp parameters
        self.init_vasp_sampling_parameters(npt=False)

        for phase in self.phases:
            # The phase sampling parameters
            args = getitem(self.config, ["vasp", "sampling", "nvt", phase])
            if len(args) == 0:
                continue

            # Get the volumes. If the volumes is a string, eval it. 
            volumes = asarray_or_eval(args.get("volumes", []))
            if len(volumes) == 0:
                continue
            size = len(volumes)

            # Get the temperatures. 
            vt_method = args.get("vt_method", "pair")
            t0, t1 = self._get_temperatures(args, vt_method, size)

            # Make (V, T) grid if needed
            if vt_method == "grid":
                _, t0 = np.meshgrid(volumes, t0)
                volumes, t1 = np.meshgrid(volumes, t1)
                volumes = volumes.flatten()
                t0 = t0.flatten()
                t1 = t1.flatten()

            # Get the number of steps
            steps = args.get("nsteps", 5000)

            # Loop through the volumes
            for i, vol in enumerate(volumes):
                jobdir = workdir / phase / f"nvt/v{np.round(vol*100, 0):.0f}"
                jobdir.mkdir(parents=True, exist_ok=True)

                # Get the supercell structures
                supercells = self.get_supercells_at_volume(phase, vol)
                if len(supercells) == 0:
                    continue

                for supercell in supercells:
                    # Determine the task id and name
                    taskid, taskname = self._get_nvt_sampling_task(
                        jobdir, t0[i], t1[i], len(supercell), override
                    )
                    if taskid < 0:
                        continue
                    # setup the Vasp calculator and generate input files.
                    taskdir = jobdir / taskname
                    taskdir.mkdir(exist_ok=True)
                    self.vasp.set(directory=str(taskdir))
                    self.vasp.set(tebeg=t0[i], teend=t1[i], nsw=steps)
                    if self.is_finite_temperature:
                        avg_temp = (t0[i] + t1[i]) / 2
                        self.vasp.set(sigma=avg_temp * kB, ismear=-1)
                    nbands = self.vasp_nbands["sampling"]
                    if nbands is not None:
                        if isinstance(nbands, dict):
                            self.vasp.set(nbands=nbands[str(len(supercell))])
                        else:
                            self.vasp.set(nbands=nbands)
                    self.vasp.write_input(supercell)
                    metadata = {
                        "phase": str(phase),
                        "ensemble": "nvt",
                        "P/V": f"v{vol:.2f}",
                        "Tstart": int(t0[i]),
                        "Tstop": int(t1[i]),
                        "nsw": int(steps),
                    }
                    with open(taskdir / "metadata.json", "w") as fp:
                        json.dump(metadata, fp, indent=2)
                        fp.write("\n")
                    print(f"[VASP/nvt/sampling/gen]: {taskdir}")
                    batch_jobs.append(str(taskdir.relative_to('sampling')))
        with open(workdir / "batch_jobs", "a") as fp:
            fp.write("\n".join(batch_jobs) + "\n")

    @staticmethod
    def _get_npt_sampling_task(
        jobdir: Path, t0: float, t1: float, v0: float, natoms: int, override=False
    ):
        """
        Returns (id, name) for the NVT sampling task.
        If the returned id is -1, the task should be skipped.
        """
        vkey = f"v{np.round(v0*100, 0):.0f}"
        files = [x.name for x in jobdir.glob(f"*_{natoms}atoms_{vkey}_*K_to_*K")]
        key = f"{natoms}atoms_{vkey}_{t0:.0f}K_to_{t1:.0f}K"
        exists = False
        for afile in files:
            if afile.endswith(key):
                exists = True
                break
        if exists and not override:
            return -1, None
        if override:
            newid = len(files)
        else:
            newid = len(files) + 1
        return newid, f"{newid}_{key}"

    def create_vasp_sampling_npt_tasks(self, override=False):
        """
        Create VASP Parrinello-Rahman NPT sampling jobs
        """
        workdir = self.root / "sampling"
        workdir.mkdir(exist_ok=True)
        batch_jobs = []

        # The NPT vasp parameters
        self.setup_vasp_sampling_parameters(npt=True)

        for phase in self.phases:
            # The phase sampling parameters
            args = getitem(self.config, ["vasp", "sampling", "npt", phase])
            if len(args) == 0:
                continue

            # Get desired pressures
            pressures = asarray_or_eval(args.get("pressures", []))
            if len(pressures) == 0:
                continue
            size = len(pressures)

            # Get temperatures
            volumes = scalar2array(asarray_or_eval(args["volumes"]), size, 
                                   n1="volumes", n2="pressures")
            t0 = scalar2array(asarray_or_eval(args["tstart"]), size, 
                              n1="tstart", n2="pressures")
            t1 = scalar2array(asarray_or_eval(args["tstop"]), size, 
                              n1="tstop", n2="pressures")
            steps = args.get("nsteps", 5000)

            # Loop through the pressures
            for i, pressure in enumerate(pressures):
                jobdir = workdir / phase / f"npt/{pressure:.0f}GPa"
                jobdir.mkdir(parents=True, exist_ok=True)

                # Get the supercell structures
                supercells = self.get_supercells_at_volume(phase, volumes[i])
                if len(supercells) == 0:
                    continue

                for supercell in supercells:
                    # Determine the task id and name
                    taskid, taskname = self._get_npt_sampling_task(
                        jobdir, t0[i], t1[i], volumes[i], len(supercell), override
                    )
                    if taskid < 0:
                        continue
                    # setup the Vasp calculator and generate input files.
                    taskdir = jobdir / taskname
                    taskdir.mkdir(exist_ok=True)
                    self.vasp.set(directory=str(taskdir))
                    self.vasp.set(tebeg=t0[i], teend=t1[i], nsw=steps)
                    self.vasp.set(pstress=pressure * 10)
                    if self.is_finite_temperature:
                        avg_temp = (t0[i] + t1[i]) / 2
                        self.vasp.set(sigma=avg_temp * kB)
                    self.vasp.write_input(supercell)
                    metadata = {
                        "phase": str(phase),
                        "ensemble": "npt",
                        "P/V": f"{pressure}GPa",
                        "Tstart": int(t0[i]),
                        "Tstop": int(t1[i]),
                        "nsw": int(steps),
                    }
                    with open(taskdir / "metadata.json", "w") as fp:
                        json.dump(metadata, fp, indent=2)
                        fp.write("\n")
                    print(f"[VASP/npt/sampling/gen]: {taskdir}")
                    batch_jobs.append(str(taskdir.relative_to('sampling')))
        with open(workdir / "batch_jobs", "a") as fp:
            fp.write("\n".join(batch_jobs) + "\n")

    # --------------------------------------------------------------------------
    # Check the status of sampling jobs
    # --------------------------------------------------------------------------

    @staticmethod
    def _is_sampling_job_finished(jobdir: Path):
        """
        Return True if the sampling job is finished.
        """
        metadata = jobdir / "metadata.json"
        if not metadata.exists():
            return False
        with open(metadata, "r") as fp:
            status = json.load(fp)
        if status.get("SU", -1) <= 0:
            return False
        return True

    def update_status_of_sampling_job(self, jobdir: Path):
        """
        Update and return the status of a sampling job at given jobdir.
        """
        metadata = jobdir / "metadata.json"
        if not metadata.exists():
            return {}

        with open(metadata, "r") as fp:
            status = json.load(fp)

        status["nrun"] = -1
        status["device"] = ""
        status["SU"] = -1
        status["processed"] = "n"

        oszicar = jobdir / "OSZICAR"
        if not oszicar.exists():
            return status

        cmd = f"grep '.*T=.*E=.*' {oszicar} | tail -n 1"
        p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        stdout, _ = p.communicate()
        if p.returncode != 0:
            return status

        nrun = int(stdout.decode().split()[0])
        status["nrun"] = nrun

        # The simulation is not finished
        if nrun < status["nsw"]:
            return status

        # The simulation is finished
        job = VaspJob(jobdir)
        su = job.get_service_unit()
        status["device"] = su.device
        status["SU"] = np.round(su.hours, 2)
        with open(metadata, "w") as fp:
            json.dump(status, fp, indent=2)
            fp.write("\n")

        # The job is already postprocessed.
        output_file = jobdir / "trajectory.extxyz"
        if output_file.exists():
            status["processed"] = "y"

        return status

    def get_status_of_all_sampling_jobs(self):
        """
        Get the status of all sampling jobs.
        """
        status = {}
        for task in self.sampling_task_iterator():
            info = self.update_status_of_sampling_job(task)
            if len(info) == 0:
                continue
            for key, value in info.items():
                status[key] = status.get(key, []) + [
                    value,
                ]
        df = pd.DataFrame(status)
        df.set_index("phase", inplace=True)
        print(df.to_string())
        with open(self.root / "sampling" / "status", "w") as fp:
            fp.write("# " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            df.to_string(fp)

    def list_unsubmitted_sampling_jobs(self):
        """
        List unsubmitted AIMD sampling jobs.
        """
        for task in self.sampling_task_iterator():
            metadata = task / "metadata.json"
            if not metadata.exists():
                print(task)
            status = self.update_status_of_sampling_job(task)
            if status.get("nrun", -1) < 0:
                print(task)
