#!/bin/bash
# This script is used to test the sampling capability of TensorDB.

#SBATCH --job-name=sampling
#SBATCH --output=output.sampling
#SBATCH --error=error.sampling
#SBATCH --partition=8gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

module purge
module load vasp-gpu/6.5.0

# OpenACC + OpenMP
export MKL_THREADLING_LAYER=INTEL
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$OMP_NUM_THREADS
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

for jobid in `seq 1 1 2`; do
    cd sampling
    jobdir=$(sed -n "${jobid}p" batch_jobs)
    cd $jobdir
    echo "Running VASP in $jobdir, `date`"
    echo "nprocs=$nprocs, pps=$pps, OMP_NUM_THREADS=$OMP_NUM_THREADS"
    mpirun -np $nprocs --map-by ppr:$pps:socket:PE=$OMP_NUM_THREADS --bind-to core vasp_gam
    sleep 1
    cd $workdir
done
echo "Done, `date`"
