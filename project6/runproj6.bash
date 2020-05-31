#!/bin/bash

#************************************************
 # Author: Zachary Reed
 # Description: Project 6 bash runner
 # Date: 5/31/2020
#************************************************

#SBATCH -J proj6
#SBATCH -A cs475-575
#SBATCH -p class
#SBATCH --gres=gpu:1
#SBATCH -o proj6.out
#SBATCH -e proj6.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=reedz@oregonstate.edu

# local work size, divisible by 32 to fill warp

for part in {1..3}
do  
    echo "--------------------------------------------"
    echo "Part $part"
    echo "Local Size,Array Size,MegaPerformance"

    if [ "$part" -eq 3 ]; then
        for numElements in 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608
        do
            g++ -DPART=$part -DNUM_ELEMENTS=$numElements -o proj6 proj6.cpp /usr/local/apps/cuda/cuda-10.1/lib64/libOpenCL.so.1.1 -lm -fopenmp
            ./proj6
        done

    else
        for localS in 8 16 32 64 128 256 512
        do  
            # for 1KiB, 2KiB, 4KiB, 8KiB, 16KiB, 32KiB, 64KiB, 128KiB, 256KiB, 512KiB, 1MiB, 2 MiB, 4MiB, 8MiB
            for numElements in 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608
            do
                g++ -DLOCAL_SIZE=$localS -DPART=$part -DNUM_ELEMENTS=$numElements -o proj6 proj6.cpp /usr/local/apps/cuda/cuda-10.1/lib64/libOpenCL.so.1.1 -lm -fopenmp
                ./proj6
            done
        done
    fi
done