#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This module defines calculators, which are used to execute high-precision DFT 
# calculations.

import datetime
import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
from collections import Counter
from typing import List
from ase import Atoms
from ase.calculators.vasp.vasp import Vasp
from ase.io import read, write
from ase.units import kB
from tensordb.utils import getitem
from tensordb.vaspkit import VaspJob, ServiceUnit
from tensordb.sampler import *


class BaseCalculator:
    """
    The base class for all high-precision DFT calculators.
    """
    
    def __init__(self, root: Path, config: dict):
        self.root = root
        self.config = config
        self.species = config["species"]
        self.workdir: Path = None
        self.init_ab_initio_calculator()
    
    @property
    def software(self):
        """
        Return the associated software.
        """
        return ""
    
    def purge(self):
        """
        Purge the high-precision DFT calculation jobs.
        """
        if isinstance(self.workdir, Path) and self.workdir.exists():
            self.workdir.rmtree(ignore_errors=True)
    
    def init_ab_initio_calculator(self):
        """
        Initialize the base calculator.
        """
        raise NotImplementedError("init_ab_initio_calculator() is not implemented.")
    
    def task_iterator(self):
        """
        Iterate through all high-precision DFT calculation job dirs.
        """
        raise NotImplementedError("task_iterator() is not implemented.")
    
    def create_tasks(self, samplers: List[BaseSampler], **kwargs):
        """
        Create high precision DFT calculation tasks.
        """
        raise NotImplementedError("create_tasks() is not implemented.")
    
    def update_status(self):
        """
        Update the status of all high precision DFT calculations.
        """
        raise NotImplementedError("update_status() is not implemented.")
    
    def get_atoms(self, taskdir: Path, check_job_status=False) -> Atoms:
        """
        Return the `Atoms` object from a high precision DFT calculation.
        """
        raise NotImplementedError("get_atoms() is not implemented.")


class VaspCalculator(BaseCalculator):
    """
    The high-precision DFT calculator class for VASP.
    """

    def __init__(self, root: Path, config: dict):
        super().__init__(root, config)
        self.workdir = self.root / "calc"
    
    @property
    def software(self):
        return "vasp"
    
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

        # Initialize the VASP calculator for high-precision DFT calculations.
        params = getitem(self.config, ["vasp", "calc"])
        self.vasp = Vasp(
            xc=params.get("xc", "pbe"),
            setups=setups,
            ediff=params.get("ediff", 1e-6),
            lreal=params.get("lreal", False),
            kspacing=params.get("kspacing", 0.2),
            prec=params.get("prec", "Accurate"),
            encut=params.get("encut", 500),
            ismear=params.get("ismear", 1),
            sigma=params.get("sigma", 0.05),
            algo=params.get("algo", "normal"),
            isym=params.get("isym", 0),
            nelmin=params.get("nelmin", 4),
            isif=params.get("isif", 2),
            ibrion=params.get("ibrion", -1),
            nsw=params.get("nsw", 1),
            nwrite=params.get("nwrite", 1),
            lcharg=params.get("lcharg", False),
            lwave=params.get("lwave", False),
            nblock=params.get("nblock", 1),
        )
        if "npar" in params:
            self.vasp.set(npar=params["npar"])
        if "kpar" in params:
            self.vasp.set(kpar=params["kpar"])
        if "ncore" in params:
            self.vasp.set(ncore=params["ncore"])
        self.vasp_nbands = params.get("nbands", None)
    
    def task_iterator(self):
        """
        Iterate through all high-precision DFT calculation job dirs.
        """
        return self.workdir.glob("*atoms/group*/task*")

    # --------------------------------------------------------------------------
    # High-precision DFT calculations
    # --------------------------------------------------------------------------

    def setup_vasp_accurate_dft_parameters(self, atoms):
        """
        Setup the parameters for high-precision DFT calculations.

        ASE doe not implement the non-collinear settings, so we should hack it.
        """
        params = getitem(self.config, ["vasp", "calc"])
        magmon_orig_value = None
        for key, value in params.items():
            if key == "magmom":
                magmon_orig_value = value
            else:
                self.vasp.set(**{key: value})
        if magmon_orig_value is not None:
            if self.vasp.bool_params["lsorbit"]:
                magmom = f"{len(atoms)*3}*{magmon_orig_value}"
            else:
                magmom = f"{len(atoms)}*{magmon_orig_value}"
        else:
            magmom = None
        if self.config.get("finite_temperature", False):
            self.vasp.set(sigma=atoms.info['etemperature'])
            self.vasp.set(ismear=-1)
        nbands = self.vasp_nbands
        if nbands is not None:
            if isinstance(nbands, str) and nbands.startswith("lambda"):
                a = atoms
                n = len(a)
                v = a.get_volume() / n
                t = atoms.info['etemperature']
                nval = eval(nbands)(a, n, v, t)
                self.vasp.set(nbands=nval)
            elif isinstance(nbands, dict):
                self.vasp.set(nbands=nbands[str(len(atoms))])
            else:
                self.vasp.set(nbands=nbands)
        return {"MAGMOM": magmom}

    def create_tasks(self, samplers: List[BaseSampler], **kwargs):
        """
        Create VASP high precision DFT calculation tasks.
        """
        self.workdir.mkdir(exist_ok=True)

        # The global hash table
        hash_file = self.workdir / "hash.json"

        # The global training structures file
        calc_file = self.workdir / "accurate_dft_calc.extxyz"

        # May read existing results
        if hash_file.exists():
            with open(hash_file, "r") as fp:
                hash_table = json.load(fp)
            calc_list = read(calc_file, index=":")
            if len(calc_list) != len(hash_table):
                raise IOError(
                    f"{calc_file}(n={len(calc_list)}) does not "
                    f"match with {hash_file}(n={len(hash_table)})!"
                )
        else:
            hash_table = {}
            calc_list = []

        # Initialize the subsets
        subset_id = Counter()
        for atoms in calc_list:
            subset_id[len(atoms)] += 1

        # Loop through all sampling tasks
        for sampler in samplers:
            for task in sampler.task_iterator():
                selected = sampler.get_samples(task, **kwargs)
                for atoms in selected:
                    hash_id = atoms.info["_hash"]
                    src = atoms.info["_source"]
                    if hash_id in hash_table:
                        continue
                    else:
                        calc_list.append(atoms)
                        natoms = len(atoms)
                        aid = f"{natoms}.{subset_id[natoms]}"
                        hash_table[hash_id] = {"aid": aid, "source": src}
                        subset_id[len(atoms)] += 1

        # Save the hash table
        with open(hash_file, "w") as fp:
            json.dump(hash_table, fp, indent=2)
            fp.write("\n")

        # Save the structures
        write(calc_file, calc_list, format="extxyz")

        # Create VASP jobs
        subset_size = Counter()
        for atoms in calc_list:
            # The original structure id
            aid = hash_table[atoms.info["_hash"]]["aid"]

            # For structures of different sizes, we may use different CPU/GPU
            # settings. Hence, 'natoms' is the first metric for makeing subsets.
            natoms = len(atoms)
            subsetdir = self.workdir / f"{natoms}atoms"
            subsetdir.mkdir(exist_ok=True)
            if natoms not in subset_size:
                subset_size[natoms] = Counter()

            # The group id. Each group contains 100 structures at most.
            sid = int(aid.split(".")[1])
            group_id = sid // 100
            groupdir = subsetdir / f"group{group_id}"
            groupdir.mkdir(exist_ok=True)

            # The task id.
            task_id = sid % 100
            taskdir = groupdir / f"task{task_id}"
            taskdir.mkdir(exist_ok=True)

            # Setup the VASP calculator
            magmom_dct = self.setup_vasp_accurate_dft_parameters(atoms)

            # Write the input files
            self.vasp.set(directory=str(taskdir))
            self.vasp.write_input(atoms)

            # Hack the MAGMOM tag for non-collinear calculations
            if magmom_dct is not None:
                with open(taskdir / "INCAR", "a") as fp:
                    fp.write("\n")
                    for key, value in magmom_dct.items():
                        if value is not None:
                            fp.write(f" {key} = {value}\n")

            subset_size[natoms][group_id] += 1

            # Write the metadata
            metadata = {
                "source": atoms.info["_source"],
                "hash": atoms.info["_hash"],
                "aid": aid,
                "group_id": group_id,
                "task_id": task_id,
            }
            if self.config.get("finite_temperature", False):
                metadata["etemperature(K)"] = atoms.info["etemperature"] / kB
            with open(taskdir / "metadata.json", "w") as fp:
                json.dump(metadata, fp, indent=2)
                fp.write("\n")

        for natoms, group_size in subset_size.items():
            for group_id, size in group_size.items():
                print(
                    f"[VASP/calc/create/{natoms}]: " f"group{group_id} ({size} tasks)"
                )

    def update_status(self):
        """
        Get the status of high precision dft calculations.
        """

        # Determine the total number of prepared jobs
        hash_file = self.root / "calc" / "hash.json"
        if not hash_file.exists():
            return
        with open(hash_file) as fp:
            hash_table = json.load(fp)
        subset_size = Counter()
        for key, value in hash_table.items():
            aid = value["aid"]
            i, j = [int(x) for x in aid.split(".")]
            subset_size[i] = max(subset_size[i], j + 1)

        status = {
            "group": [],
            "total_jobs": [],
            "completed_jobs": [],
            "converged_jobs": [],
            "CPU(jobs)": [],
            "CPU(hours)": [],
            "GPU(jobs)": [],
            "GPU(hours)": [],
        }
        accumulator = {}

        # Loop through all prepared jobs
        for taskdir in self.task_iterator():
            metadata = taskdir / "metadata.json"
            if not metadata.exists():
                continue
            with open(metadata, "r") as fp:
                metadata = json.load(fp)
            sid, aid = [int(x) for x in metadata["aid"].split(".")]
            gid = metadata["group_id"]
            key = (sid, gid)
            if key not in accumulator:
                if (gid + 1) * 100 <= subset_size[sid]:
                    n_total = 100
                else:
                    n_total = aid % 100
                accumulator[key] = Counter(
                    {
                        "CPU(hours)": 0.0,
                        "GPU(hours)": 0.0,
                        "CPU(jobs)": 0,
                        "GPU(jobs)": 0,
                        "n_converged": 0,
                        "n_total": n_total,
                        "completed_tasks": [],
                        "converged_tasks": [],
                    }
                )
            job = VaspJob(taskdir)
            su = job.get_vasp_job_service_unit()
            if su is None:
                continue
            converged = job.check_vasp_job_scf_convergence()
            if converged:
                accumulator[key]["n_converged"] += 1
                accumulator[key]["converged_tasks"].append(str(taskdir))
            accumulator[key]["n_completed"] += 1
            accumulator[key]["completed_tasks"].append(str(taskdir))
            if su.device == "cpu":
                accumulator[key]["CPU(hours)"] += su.hours
                accumulator[key]["CPU(jobs)"] += 1
            else:
                accumulator[key]["GPU(hours)"] += su.hours
                accumulator[key]["GPU(jobs)"] += 1
            # Update the metadata
            with open(taskdir / "metadata.json", "w") as fp:
                metadata["SU"] = su.__dict__
                metadata["converged"] = converged
                json.dump(metadata, fp, indent=2)
                fp.write("\n")

        # Save the group metadata
        for key in accumulator:
            sid, gid = key
            groupdir = Path(f"calc/{sid}atoms/group{gid}")
            with open(groupdir / "metadata.json", "w") as fp:
                json.dump(accumulator[key], fp, indent=2)
                fp.write("\n")

        # Print the status
        for key, value in accumulator.items():
            status["group"].append(f"{key[0]}.g{key[1]}")
            status["total_jobs"].append(value["n_total"])
            status["converged_jobs"].append(value["n_converged"])
            status["completed_jobs"].append(value["n_completed"])
            status["CPU(jobs)"].append(value["CPU(jobs)"])
            status["GPU(jobs)"].append(value["GPU(jobs)"])
            status["CPU(hours)"].append(np.round(value["CPU(hours)"], 2))
            status["GPU(hours)"].append(np.round(value["GPU(hours)"], 2))
        status["group"].append("overall")
        status["total_jobs"].append(sum(status["total_jobs"]))
        status["CPU(jobs)"].append(sum(status["CPU(jobs)"]))
        status["GPU(jobs)"].append(sum(status["GPU(jobs)"]))
        status["CPU(hours)"].append(sum(status["CPU(hours)"]))
        status["GPU(hours)"].append(sum(status["GPU(hours)"]))
        status["completed_jobs"].append(sum(status["completed_jobs"]))
        status["converged_jobs"].append(sum(status["converged_jobs"]))
        df = pd.DataFrame(status)
        df.set_index("group", inplace=True)
        print(df.to_string())
        with open(self.root / "calc" / "status", "w") as fp:
            fp.write("# " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            df.to_string(fp)
    
    def get_atoms(self, taskdir: Path, check_job_status=False):
        """
        Gather one completed high precision DFT calculation.
        """
        if check_job_status:
            job = VaspJob(taskdir)
            su = job.get_vasp_job_service_unit()
            if su is None:
                return None
            converged = job.check_vasp_job_scf_convergence()
            if not converged:
                return None
        else:
            metadata = taskdir / "metadata.json"
            if not metadata.exists():
                return None
            with open(metadata, "r") as fp:
                metadata = json.load(fp)
            if not metadata.get("converged", False):
                return None
            if not metadata.get("SU", {}):
                return None
            su = ServiceUnit(**metadata["SU"])
            if su.hours <= 0:
                return None
        atoms = VaspJob(taskdir).get_atoms(
            index=-1, 
            is_finite_temperature=self.config.get("finite_temperature", False))
        atoms.info["hash"] = metadata["hash"]
        atoms.info["aid"] = metadata["aid"]
        atoms.info["group_id"] = metadata["group_id"]
        atoms.info["task_id"] = metadata["task_id"]
        return atoms
