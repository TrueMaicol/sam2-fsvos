#!/bin/bash

# # $1 = dataset path
# # $2 = output dit
# # $3 = fold
# # $4 = run number

#SBATCH --time=08:00:00                         # Maximum wall time (hh:mm:ss)
#SBATCH --nodes=1                               # Number of nodes to use
#SBATCH --ntasks-per-node=1                     # Number of MPI tasks per node (e.g., 1 per GPU)
#SBATCH --cpus-per-task=1                      # Number of CPU cores per task (adjust as needed)
#SBATCH --gres=gpu:1                            # Number of GPUs per node (adjust to match hardware)
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=iscrc_marsv2                  # Project account number

# Load necessary modules (adjust to your environment)
module load profile/deeplrn
module load cineca-ai/4.3.0

source ./sam2_venv/bin/activate
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"

cd src

if [ -n "$4" ]; then
    run_dir="FULL_TEST_$4"
    run_n=$4
else
    run_dir="FULL_TEST_1"
    run_n=1
fi

if [ -n "$5" ]; then
    N_SHOT=$5
else
    N_SHOT=5
fi

srun python test_SAM2_FSVOS.py \
    --benchmark youtube-fsvos \
    --checkpoint sam2.1_hiera_tiny.pt \
    --config sam2.1/sam2.1_hiera_t.yaml \
    --fold ${2} \
    --data_list_path /leonardo_work/IscrC_MARSv2/datasets/VSPW_480p/lists/test.txt \
    --dataset_path /leonardo_work/IscrC_MARSv2/datasets/VSPW_480p/data \
    --output_dir /leonardo_scratch/large/userexternal/mcavicch/SAM2_OUTPUT_DATA/${1}/${run_dir} \
    --session_name fold_${2} \
    --seed ${3} \
    --nshot ${N_SHOT} \
    --run_number ${run_n} \
    --benchmark minivspw \
    --random_state_path /leonardo_work/IscrC_MARSv2/SAM2_FSVOS/src/minivspw_random_state/VSPW/SAM3_GEN_LABEL_SUPPORT/${N_SHOT}-SHOT \