#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This module defines samplers, which are used to generate structures for high-precision 
# DFT calculations.

import numpy as np
import pandas as pd
import os
import json
import hashlib
import shutil
from datetime import datetime
from subprocess import Popen, PIPE
from ase import Atoms
from ase.units import kB
from ase.calculators.vasp import Vasp
from ase.build import bulk
from ase.io import write, read
from pathlib import Path
from tensordb.utils import getitem, asarray_or_eval, scalar2array
from tensordb.vaspkit import VaspJob
from tensoralloy.io.vasp import read_vasp_xml


__all__ = ["BaseSampler", "AimdSampler", "VaspAimdSampler"]


class BaseSampler:
    """
    The base class for all types of samplers.
    A sampler is used to generate structures for high accuracy DFT calculations.
    """

    def __init__(self, root, config: dict):
        """
        Initialize the base sampler.
        """
        self.root = root
        self.config = config
        self.species = self.config["species"]
        self.phases = self.config["phases"]
        self.base_phase_structures = self.init_phases()
        self.workdir = None
        self.init_ab_initio_calculator()
    
    def purge(self):
        """
        Purge the sampling directory.
        """
        if isinstance(self.workdir, Path) and self.workdir.exists():
            shutil.rmtree(self.workdir, ignore_errors=True)

    # --------------------------------------------------------------------------
    # Initialization methods
    # 1. init_phases: initialize the base phase structures.
    # 2. init_liquid_structure: initialize the liquid phase structure.
    # 3. get_base_structure: get the base phase structure.
    # 4. get_supercells_at_volume: get the supercell structure at the given volume.
    # 5. init_ab_initio_calculator: initialize the ab initio calculator.
    # --------------------------------------------------------------------------
    
    def init_phases(self):
        """
        Initialize the base phase structures.
        """
        workdir = self.root / "structures"
        if not workdir.exists():
            raise IOError(f"Cannot find the directory 'structures' in {str(self.root)}")
        structures = {}
        for phase in self.config["phases"]:
            if phase == "liquid":
                structures[phase] = self.init_liquid_structure()
            else:
                candidates = [x for x in workdir.glob(f"{phase}.*")]
                if len(candidates) == 0:
                    raise ValueError(f"Cannot find the poscar for {phase}")
                if len(candidates) > 1:
                    raise ValueError(f"Multiple poscars for {phase}")
                poscar = candidates[0]
                structures[phase] = read(str(poscar))
        return structures

    def init_liquid_structure(self):
        """
        Initialize the liquid phase structure.

        Num species = 1: use the fcc phase.
        Num species > 1: use SAE (todo)

        """
        if len(self.species) == 1:
            veq = self.config["liquid"]["veq"]
            a = (4 * veq) ** (1 / 3)
            return bulk(self.species[0], crystalstructure="fcc", a=a, cubic=True)
        else:
            raise NotImplementedError(
                "Liquid phase for multi-species is not implemented yet."
            )

    def get_base_structure(self, phase: str) -> Atoms:
        """
        Get the base phase structure.
        """
        return self.base_phase_structures[phase].copy()

    def get_supercells_at_volume(self, phase: str, volume: float) -> Atoms:
        """
        Get the supercell structure at the given atomic volume.
        """
        base = self.get_base_structure(phase)
        scale = (volume / base.get_volume() * len(base)) ** (1 / 3)
        base.set_cell(base.get_cell() * scale, scale_atoms=True)
        supercells = []
        for replicate in self.config[phase]["supercell"]:
            supercells.append(base * replicate)
        return supercells
    
    def init_ab_initio_calculator(self):
        """
        Initialize the ab initio calculator.
        """
        raise NotImplementedError("init_ab_initio_calculator() is not implemented yet.")

    # --------------------------------------------------------------------------
    # Task manupulation methods
    # 1. task_iterator: iterate through all tasks.
    # 2. create_tasks: create all tasks.
    # 3. is_task_finished: check if the task is finished.
    # 4. update_status_of_task: update the status of a task.
    # 5. update_status: update the status of all tasks.
    # 6. list_unsubmitted_tasks: list unsubmitted tasks.
    # 7. post_process_task: post-process a task.
    # 8. post_process: post-process all tasks.
    # --------------------------------------------------------------------------
    
    def task_iterator(self):
        """
        Iterate through all sampling tasks.
        """
        raise NotImplementedError("task_iterator() is not implemented yet.")
    
    def create_tasks(self, override=False):
        """
        Create all sampling tasks.
        """
        raise NotImplementedError("create_tasks() is not implemented yet.")
    
    @staticmethod
    def is_task_finished(taskdir: Path):
        """
        Return True if the task is finished.
        """
        metadata = taskdir / "metadata.json"
        if not metadata.exists():
            return False
        with open(metadata, "r") as fp:
            status = json.load(fp)
        if status.get("SU", -1) <= 0:
            return False
        return True
    
    def update_status_of_task(self, taskdir: Path):
        """
        Update the status of task at `taskdir`.
        """
        raise NotImplementedError("update_status_of_task() is not implemented yet.")

    def update_status(self):
        """
        Update the status of all tasks.
        """
        status = {}
        for task in self.task_iterator():
            info = self.update_status_of_task(task)
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

    def list_unsubmitted_tasks(self):
        """
        List unsubmitted tasks.
        """
        for task in self.task_iterator():
            metadata = task / "metadata.json"
            if not metadata.exists():
                print(task)
            status = self.update_status_of_task(task)
            if status.get("nrun", -1) < 0:
                print(task)
    
    def post_process_task(self, jobdir: Path):
        """
        Post-process a task.
        """
        raise NotImplementedError("post_process_a_task() is not implemented yet.")
    
    def post_process(self):
        """
        Post-processing all sampling jobs.
        """
        for task in self.task_iterator():
            self.post_process_task(task)
    
    # --------------------------------------------------------------------------
    # Sample generation methods
    # 1. get_samples: get samples from a task.
    # --------------------------------------------------------------------------

    def get_samples(self, task: Path, interval=50, **kwargs):
        raise NotImplementedError("get_samples() is not implemented yet.")


class AimdSampler(BaseSampler):
    """
    The base class for ab initio molecular dynamics (AIMD) based samplers.
    """

    TRAJECTORY_FILE = "trajectory.extxyz"
    
    def __init__(self, root, config):
        """
        Initialize the AIMD sampler. 
        The workdir is <root>/sampling for all AIMD samplers.
        """
        super().__init__(root, config)
        self.workdir = self.root / "sampling"
    
    def task_iterator(self):
        """
        Iterate through all AIMD sampling tasks.
        """
        return self.workdir.glob("*/n[pv]t/*/*_*K_to_*K")
    
    def create_tasks(self, override=False):
        """
        Create all VASP AIMD sampling tasks.
        """
        self.create_aimd_nvt_tasks(override=override)
        self.create_aimd_npt_tasks(override=override)
    
    @staticmethod
    def get_nvt_sampling_task(jobdir: Path, t0: float, t1: float, natoms: int, 
                              override=False):
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
    def get_temperatures(args: dict, vt_method: str, size: int, npt=False):
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

    def create_aimd_nvt_tasks(self, override=False):
        """
        Create VASP Langevin NVT sampling jobs.
        """
        raise NotImplementedError("create_aimd_nvt_tasks() is not implemented yet.")
    
    @staticmethod
    def get_npt_sampling_task(jobdir: Path, t0: float, t1: float, v0: float, 
                              natoms: int, override=False):
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
    
    def create_aimd_npt_tasks(self, override=False):
        """
        Create VASP Parrinello-Rahman NPT sampling jobs.
        """
        raise NotImplementedError("create_aimd_npt_tasks() is not implemented yet.")
    
    def get_samples(self, task: Path, interval=50, shuffle=False, **kwargs):
        """
        Get samples from the AIMD sampling tasks.
        """
        selected = []
        # Check if the trajectory file exists
        trajectory = task / self.TRAJECTORY_FILE
        if not trajectory.exists():
            return selected
        # Read the trajectory file
        full = read(trajectory, index=slice(0, None, 1))
        if shuffle:
            size = len(full) // interval
            selected = np.random.choice(
                full[interval - 1 :], size=size, replace=False
            )
        else:
            # Select the last snapshot of each block(size=interval)
            selected = full[interval - 1 :: interval]
            # If the interval is larger than the length of the trajectory, select the 
            # last one.
            if not selected:
                selected = [full[-1]]
        return selected


class VaspAimdSampler(AimdSampler):
    """
    The AIMD sampler based on VASP.
    """
    
    def init_ab_initio_calculator(self):
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
        self.vasp_nbands = params.get("nbands", None)

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

    def create_aimd_nvt_tasks(self, override=False):
        """
        Create VASP Langeven NVT sampling jobs: gamma-only.
        """
        self.workdir.mkdir(exist_ok=True)
        batch_jobs = []

        # The NVT vasp parameters
        self.init_vasp_sampling_parameters(npt=False)

        for phase in self.phases:
            # The phase sampling parameters
            args = getitem(self.config, ["aimd", "sampling", "nvt", phase])
            if len(args) == 0:
                continue

            # Get the volumes. If the volumes is a string, eval it. 
            volumes = asarray_or_eval(args.get("volumes", []))
            if len(volumes) == 0:
                continue
            size = len(volumes)

            # Get the temperatures. 
            vt_method = args.get("vt_method", "pair")
            t0, t1 = self.get_temperatures(args, vt_method, size)

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
                jobdir = self.workdir / phase / f"nvt/v{np.round(vol*100, 0):.0f}"
                jobdir.mkdir(parents=True, exist_ok=True)

                # Get the supercell structures
                supercells = self.get_supercells_at_volume(phase, vol)
                if len(supercells) == 0:
                    continue

                for supercell in supercells:
                    # Determine the task id and name
                    taskid, taskname = self.get_nvt_sampling_task(
                        jobdir, t0[i], t1[i], len(supercell), override
                    )
                    if taskid < 0:
                        continue
                    # setup the Vasp calculator and generate input files.
                    taskdir = jobdir / taskname
                    taskdir.mkdir(exist_ok=True)
                    self.vasp.set(directory=str(taskdir))
                    self.vasp.set(tebeg=t0[i], teend=t1[i], nsw=steps)
                    if self.config.get("finite_temperature", False):
                        avg_temp = (t0[i] + t1[i]) / 2
                        self.vasp.set(sigma=avg_temp * kB, ismear=-1)
                    nbands = self.vasp_nbands
                    if nbands is not None:
                        if isinstance(nbands, str) and nbands.startswith("lambda"):
                            a = supercell
                            n = len(a)
                            v = a.get_volume() / n
                            t = max(t0[i], t1[i])
                            nval = eval(nbands)(a, n, v, t)
                            self.vasp.set(nbands=int(nval))
                        elif isinstance(nbands, dict):
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
                    print(f"[aimd/nvt/sampling]: {taskdir}")
                    batch_jobs.append(str(taskdir.relative_to('sampling')))
        with open(self.workdir / "batch_jobs", "a") as fp:
            fp.write("\n".join(batch_jobs) + "\n")

    def create_aimd_npt_tasks(self, override=False):
        """
        Create VASP Parrinello-Rahman NPT sampling jobs
        """
        self.workdir.mkdir(exist_ok=True)
        batch_jobs = []

        # The NPT vasp parameters
        self.init_vasp_sampling_parameters(npt=True)

        for phase in self.phases:
            # The phase sampling parameters
            args = getitem(self.config, ["aimd", "sampling", "npt", phase])
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
                jobdir = self.workdir / phase / f"npt/{pressure:.0f}GPa"
                jobdir.mkdir(parents=True, exist_ok=True)

                # Get the supercell structures
                supercells = self.get_supercells_at_volume(phase, volumes[i])
                if len(supercells) == 0:
                    continue

                for supercell in supercells:
                    # Determine the task id and name
                    taskid, taskname = self.get_npt_sampling_task(
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
                    if self.config.get("finite_temperature", False):
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
                    print(f"[aimd/npt/sampling]: {taskdir}")
                    batch_jobs.append(str(taskdir.relative_to('sampling')))
        with open(self.workdir / "batch_jobs", "a") as fp:
            fp.write("\n".join(batch_jobs) + "\n")
    
    def update_status_of_task(self, jobdir: Path):
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
        su = job.get_vasp_job_service_unit()
        status["device"] = su.device
        status["SU"] = np.round(su.hours, 2)
        with open(metadata, "w") as fp:
            json.dump(status, fp, indent=2)
            fp.write("\n")

        # The job is already postprocessed.
        output_file = jobdir / self.TRAJECTORY_FILE
        if output_file.exists():
            status["processed"] = "y"

        return status
    
    def post_process_task(self, jobdir: Path):
        """
        Post-processing a sampling job.
        """
        # Check if the job is finished
        if not self.is_task_finished(jobdir):
            return
        # Check if the vasprun.xml file exists
        vasprun_xml = jobdir / "vasprun.xml"
        if not vasprun_xml.exists():
            return
        # Check if the trajectory file exists
        output_file = jobdir / self.TRAJECTORY_FILE
        if output_file.exists():
            return
        # Read the xml file using the api `tensoralloy.io.vasp.read_vasp_xml`
        try:
            trajectory = [
                atoms
                for atoms in read_vasp_xml(
                    vasprun_xml, 
                    index=slice(0, None, 1), 
                    finite_temperature=self.config.get("finite_temperature", False)
                )
            ]
        except Exception as excp:
            print(f"[VASP/sampling/postprocess]: FAILED to read {vasprun_xml}")
            return
        if len(trajectory) == 0:
            return
        # Add the source information to the atoms.info
        for i, atoms in enumerate(trajectory):
            src = f"{str(jobdir)}@{i}"
            atoms.info["_source"] = src
            atoms.info["_hash"] = hashlib.md5(src.encode()).hexdigest()
        # Save the trajectory to an extxyz file
        write(output_file, trajectory, format="extxyz")
        print(f"[VASP/sampling/postprocess]: {jobdir}")


class ExistedTrajectorySampler(BaseSampler):
    """
    This sampler is used to sample from an existing trajectory file.
    The format should be extxyz.
    """

    def __init__(self, root, config: dict):
        """
        Initialization method.
        """
        self.root = root
        self.config = config

        # TODO: add support for finite temperature sampling using external trajectories
        if config.get("finite_temperature", False):
            raise ValueError("Currently the ExistedTrajectorySampler does not support "
                             "finite temperature sampling.")

        params = getitem(config, ["external", ])
        if "directory" not in params:
            raise ValueError(
                "The 'directory' key is not found in the 'external' section.")
        self.workdir = self.root / params["directory"]
        if not self.workdir.exists():
            raise ValueError(f"Cannot find the directory {self.workdir}")
        self.recursive = params.get("recursive", True)
    
    def task_iterator(self):
        if not self.recursive:
            return next([self.workdir, ])
        else:
            for dirpath, _, _ in os.walk(self.workdir):
                yield Path(dirpath).absolute()

    def get_samples(self, task: Path, interval=50, **kwargs):
        trajectory = []
        for afile in task.glob("*.extxyz"):
            o = read(afile, index=":")
            if isinstance(o, Atoms):
                src = f"{str(afile)}@0"
                o.info["_source"] = src
                o.info["_hash"] = hashlib.md5(src.encode()).hexdigest()
                trajectory.append(o)
            else:
                for i, atoms in enumerate(o):
                    src = f"{str(afile)}@{i}"
                    atoms.info["_source"] = src
                    atoms.info["_hash"] = hashlib.md5(src.encode()).hexdigest()
                    trajectory.append(atoms)
        selected = trajectory[::interval]
        return selected

    # --------------------------------------------------------------------------
    # Below methods are not implemented in this class.
    # --------------------------------------------------------------------------
    
    def purge(self):
        pass

    def init_phases(self):
        pass

    def init_liquid_structure(self):
        pass

    def get_base_structure(self, phase):
        pass

    def get_supercells_at_volume(self, phase, volume):
        pass

    def init_ab_initio_calculator(self):
        pass
    
    def create_tasks(self, override=False):
        pass

    @staticmethod
    def is_task_finished(self, taskdir):
        return True
    
    def update_status(self):
        pass

    def list_unsubmitted_tasks(self):
        pass

    def post_process(self):
        pass
