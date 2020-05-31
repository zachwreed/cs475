#!/bin/bash

#SBATCH -J proj6
#SBATCH -A cs475-575
#SBATCH -p class
#SBATCH --gres=gpu:1
#SBATCH -o test.out
#SBATCH -e test.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=reedz@oregonstate.edu

# local work size, divisible by 32 to fill warp

for part in {1..3}
do  

    if [ "$part" -eq 3 ]; then
        # for numElements in 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 209152 4194304 8388608
        for numElements in 1024
        do
            g++ -DPART=$part -DNUM_ELEMENTS=$numElements -o proj6 proj6.cpp /usr/local/apps/cuda/cuda-10.1/lib64/libOpenCL.so.1.1 -lm -fopenmp
            ./proj6
        done

    else
        echo "Part $part"
    fi
done