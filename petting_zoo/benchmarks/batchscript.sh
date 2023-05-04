#!/bin/bash

#SBATCH -J tf_benchmarking
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3:00:00
#SBATCH --mem=8GB
#SBATCH -o tf_no_gpu_%j.o
#SBATCH -e tf_no_gpu_%j.e
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Print key runtime properties for records
echo Master process running on `hostname`
echo Directory is `pwd`
echo Starting execution at `date`
echo Current PATH is $PATH

module load anaconda
source activate rai
python comms-tag.py

