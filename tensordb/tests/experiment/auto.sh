#!/bin/bash
# This script is used to test the entire workflow of TensorDB.

# 1. Create NVT sampling jobs.
# tensordb create sampling --nvt

# 2. Submit the NVT sampling jobs and wait for completion. It will take ~25 minutes.
# jobid=$(sbatch slurm.sampling.sh | awk '{print $4}')
# sleep 5
# failed=0
# while true; do
#     status=$(sacct -j "$jobid" --format=State --noheader | head -n1 | xargs)
#     case $status in
#         PENDING|RUNNING|EXITING)
#             echo "The job $jobid is $status. Waiting for 60 seconds."
#             sleep 60
#             ;;
#         COMPLETED)
#             echo "The job $jobid is completed."
#             break
#             ;;
#         *)
#             echo "The job $jobid is failed."
#             failed=1
#             break
#             ;;
#     esac
# done
# if [ $failed -eq 1 ]; then
#     exit 1
# fi

# 3. Postprocess the NVT sampling jobs.
tensordb status sampling
tensordb postprocess

# 4. Create high-precision dft jobs.
tensordb create calc

# srun slurm.calc.sh
# if [ -f slurm.calc.out ]; then
#     rm -f slurm.calc.out
# fi
# tensordb status calc
# tensordb purge
