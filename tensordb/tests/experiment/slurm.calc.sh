#!/bin/bash
# # This script is used to test the high-accuracy dft calculation capability of TensorDB.

#SBATCH --job-name=calc
#SBATCH --output=output.calc
#SBATCH --error=error.calc
#SBATCH --nodes=2
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1

module purge
module load vasp/6.4.2

# OpenACC + OpenMP
export MKL_THREADLING_LAYER=INTEL
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_STACKSIZE=1024m

# Two sockets per node
SOCKETS_PER_NODE=2

# Total number of processes
nprocs=$(($SLURM_JOB_NUM_NODES*$SLURM_NTASKS_PER_NODE))

# Processes per socket
pps="1"
if [ "$SLURM_NTASKS_PER_NODE" != "1" ]; then
    pps=$(($SLURM_NTASKS_PER_NODE/$SOCKETS_PER_NODE))
fi
workdir=`pwd`

declare -a jobdirs=(
    "calc/32atoms/group0/task0"
    "calc/32atoms/group0/task1"
    "calc/32atoms/group0/task2" 
    "calc/32atoms/group0/task3"
    "calc/54atoms/group0/task0"
    "calc/54atoms/group0/task1"
    "calc/54atoms/group0/task2"
    "calc/54atoms/group0/task3"
    "porous/20atoms/group0/task0"
    "porous/35atoms/group0/task0"
    "neq/32atoms/group0/task0"
    "neq/32atoms/group0/task1"
    "neq/54atoms/group0/task0"
    "neq/54atoms/group0/task1"
)

for jobdir in "${jobdirs[@]}"; do
    cd $jobdir
    echo "Running $jobdir, `date`"
    mpirun -np $nprocs vasp_std
    sleep 1
    cd $workdir
done
echo "Done, `date`"

