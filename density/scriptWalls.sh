#!/bin/bash

#$ -cwd
#$ -j yes
#$ -pe smp 16

set OMP_NUM_THREADS=3
./ContDens
