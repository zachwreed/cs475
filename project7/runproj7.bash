#!/bin/bash

#************************************************
 # Author: Zachary Reed
 # Description: Project 6 bash runner
 # Date: 5/31/2020
#************************************************

#SBATCH -J proj7
#SBATCH -A cs475-575
#SBATCH -p class
#SBATCH --gres=gpu:1
#SBATCH -o proj7.out
#SBATCH -e proj7.err
#SBATCH --mail-type=END
#SBATCH --mail-user=reedz@oregonstate.edu

for func in {0..2}
do 
    g++ -DFUNC_TYPE=$func -o proj7 proj7.cpp /usr/local/apps/cuda/cuda-10.1/lib64/libOpenCL.so.1.1 -lm -fopenmp
    ./proj7
done