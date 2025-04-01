#!/bin/bash
#SBATCH -J yolo_training
#SBATCH -o yolo_training.o%j
#SBATCH --mail-user=rdepa@uh.edu
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu

source activate /project/yzhang/rdepa/.conda/envs/myenv

# ✅ Set Conda Cache Directory
export CONDA_PKGS_DIRS=/project/yzhang/rdepa/conda_cache_dir

# ✅ Set PyTorch & YOLO Cache
export YOLO_CACHE_DIR="/project/yzhang/rdepa/cache"
mkdir -p $YOLO_CACHE_DIR

# ✅ Run Training with Correct `torchrun` Path
python -m torch.distributed.launch --nproc_per_node=8 --use_env model.py | tee log_$SLURM_JOB_ID.txt
