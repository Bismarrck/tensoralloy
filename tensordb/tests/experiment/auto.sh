#!/bin/bash
# This script is used to test the entire workflow of TensorDB.

function slurm_run {
    script=$1
    jobid=$(sbatch $script | awk '{print $4}')
    sleep 5
    failed=0
    while true; do
        status=$(sacct -j "$jobid" --format=State --noheader | head -n1 | xargs)
        case $status in
            PENDING|RUNNING|EXITING)
                echo "The job $jobid is $status. Waiting for 60 seconds."
                sleep 60
                ;;
            COMPLETED)
                echo "The job $jobid is completed."
                break
                ;;
            *)
                echo "The job $jobid is failed."
                failed=1
                break
                ;;
        esac
    done
    if [ $failed -eq 1 ]; then
        exit 1
    fi
}

# 1. Create NVT sampling jobs.
tensordb create sampling

# 2. Submit the NVT sampling jobs and wait for completion. It will take some time.
slurm_run slurm.sampling.sh

# 3. Postprocess the NVT sampling jobs.
tensordb status sampling
tensordb postprocess

# 4. Create high-precision dft jobs.
tensordb create calc
tensordb create porosity
tensordb create neq

# 5. Submit the high-precision dft jobs and wait for completion. It will take some time.
slurm_run slurm.calc.sh

# 6. Show the status of the high-precision dft jobs.
tensordb status calc
