#!/bin/bash

#SBATCH --partition=medium
#SBATCH --cpus-per-task=16
#SBATCH --mem=200MB

# #$SLURM_CPUS_PER_TASK variable

# "name" and "dirout" are named according to the testcase
# 11/24 - Updated for Cropdiversity
# 12/07 - Updated with PyStats for Multicore (d17c)

name=A6-p5C0
casename=${name}
dirout=${name}_out
options=sv
n_avg=4
smooth=1 

# "executables" are renamed and called from their directory
dirbin=../../root_bin
gencase="${dirbin}/GenCase4_linux64"
dualsphysicscpu="${dirbin}/RootSPH37c"
partvtk="${dirbin}/PartVTK4_linux64"
createVtk="${dirbin}/pyvtker_v21.py"
pyStats="${dirbin}/pystat_d17c.py"


# Library path must be indicated properly
current=$(pwd)
cd $dirbin
path_so=$(pwd)
cd $current
export LD_LIBRARY_PATH=$path_so


# "dirout" is created to store results or it is cleaned if it already exists

if [ ! -e $dirout ]; then
  mkdir $dirout
fi
diroutdata=${dirout}/data;
if [ -e $diroutdata ]; then
  rm $diroutdata/*.bi4
  rm $diroutdata/*.obi4
fi 
mkdir $diroutdata


# CODES are executed according the selected parameters of execution in this testcase
errcode=0

# Executes GenCase4 to create initial files for simulation.
if [ $errcode -eq 0 ]; then
  $gencase Def $dirout/$name -save:all
  errcode=$?
fi

# Executes DualSPHysics to simulate SPH method.
if [ $errcode -eq 0 ]; then
  $dualsphysicscpu -cpu $dirout/$name $dirout -dirdataout data -svres -sv:binx -ompthreads:$SLURM_CPUS_PER_TASK
  errcode=$?
  echo errcode
fi

# Executes PartVTK4 to create VTK files with particles.
dirout2=${dirout}/particles; 
dirimg=${dirout}/img;
if [ ! -e $dirout2 ]; then
  mkdir $dirout2
fi
if [ ! -e $dirimg ]; then
  mkdir $dirimg
fi
if [ $errcode -eq 0 ]; then
  $partvtk -threads:$SLURM_CPUS_PER_TASK -dirin $diroutdata -savecsv $dirout2/$casename -vars:Mass,Press,Qfxx,Qfyy,Qfzz,Qfyz,Qfxz,Qfxy,VonMises3D,GradVel,CellOffSpring,StrainDot,Acec,Acevisc
  
  # arguments should be separated by space, not coma nor semicolon
  python $createVtk $casename $dirout2 $smooth
  
  # Call stats to img script
  python $pyStats $casename $dirout2 $dirimg $options $n_avg
  errcode=$?
fi


if [ $errcode -eq 0 ]; then
  echo All done
else
  echo Execution aborted
fi

read -n1 -r -p "Press any key to continue..." key
echo 
