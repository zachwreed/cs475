#!/bin/bash

#************************************************
 # Author: Zachary Reed
 # Description: Project 7 bash runner
 # Date: 6/8/2020
#************************************************

#SBATCH -J proj7
#SBATCH -A cs475-575
#SBATCH -p class
#SBATCH --gres=gpu:1
#SBATCH -o proj7.out
#SBATCH -e proj7.err
#SBATCH --mail-type=END
#SBATCH --mail-user=reedz@oregonstate.edu

for func in {0..3}
do
    g++ -DFUNC_TYPE=$func proj7.cpp -o proj7 /usr/local/apps/cuda/cuda-10.1/lib64/libOpenCL.so.1.1 -lm -fopenmp
    ./proj7
done