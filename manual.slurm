#!/bin/bash
#SBATCH --job-name=in5550_home_exam
#SBATCH --account=ec403
#SBATCH --time=01:00:00
#SBATCH --partition=accel    # To use the accelerator nodes
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=15G
#SBATCH --cpus-per-task=8

# NB: this script should be run with "sbatch sample.slurm"!
# See https://www.uio.no/english/services/it/research/platforms/edu-research/help/fox/jobs/submitting.md

set -euo pipefail

module purge
module load Python/3.10.8-GCCcore-12.2.0
module load CUDA/11.7.0

# Source virtual environment
source /cluster/work/projects/ec403/ec-kjetiki/ExamIN5550/venvp310/bin/activate

# Debug (optional—remove once it works)
echo "== which python: $(which python)"
echo "== python --version: $(python --version)"


# print information (optional)
echo "submission directory: ${SUBMITDIR}"


# Run system on GPU
python manual_eval.py
