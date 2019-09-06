#!/bin/bash

#$ -cwd
#$ -j yes
#$ -pe smp 32

#set OMP_NUM_THREADS=3
./ContDens data/Simulation-2/4-HeavierOuterGruffalo 16. 360
