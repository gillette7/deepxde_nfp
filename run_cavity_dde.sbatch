#!/bin/bash
#SBATCH -A asccasc
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH -o yy_%j.out
#SBATCH -J slurm1
#SBATCH --mail-type=ALL
#SBATCH -p pbatch

cd /usr/workspace/gillette/nfp4va/deepxde_nfp
conda activate nfpLC
export PYTHONPATH=/usr/workspace/gillette/nfp4va/deepxde_nfp
python examples/pinn_forward/CavityRePsi_2D.py 