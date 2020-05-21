#!/bin/bash

#SBATCH -J proj5
#SBATCH -A cs475-575
#SBATCH -p class
#SBATCH --gres=gpu:1
#SBATCH -o proj5.out
#SBATCH -e proj5.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=reedz@oregonstate.edu

for block in 16 32 64 128
do  
    # for 16KiB, 32KiB, 64KiB, 128KiB, 256KiB, 512KiB, 1MiB
    for trial in 16384 32768 65536 131072 262144 524288 1048576
    do
        /usr/local/apps/cuda/cuda-10.1/bin/nvcc -DBLOCKSIZE=$block -DNUMTRIALS=$trial -o proj5 proj5.cu
        ./proj5
    done
done