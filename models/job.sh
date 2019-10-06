#!/bin/bash

#SBATCH --job-name=pytorch_1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=60000M
#SBATCH --partition=gpu_shared_course
#SBATCH --gres=gpu:2

module purge
module load 2019
module load eb


module load Python/3.6.6-foss-2018b
module load cuDNN/7.6.3-CUDA-10.0.130
export LD_LIBRARY_PATH=/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATH

pip3 install --user torch torchvision sklearn matplotlib tqdm torchtext spacy pytorch_transformers  keras theano

srun python3 training.py