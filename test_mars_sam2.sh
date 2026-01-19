#!/bin/bash

# # $1 = dataset path
# # $2 = output dit
# # $3 = fold

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

srun python test_SAM2_FSVOS.py \
    --checkpoint sam2.1_hiera_tiny.pt \
    --config sam2.1/sam2.1_hiera_t.yaml \
    --fold ${3} \
    --dataset_path ${1} \
    --output_dir ${2} \
    --session_name tiny_fold_${3} \
    --benchmark minivspw \
    --data_list_path /leonardo_work/IscrC_MARSv2/datasets/VSPW_480p/lists/test.txt \
