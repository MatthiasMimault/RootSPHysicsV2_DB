@echo off
REM "name" and "dirout" are named according to the testcase

set name=DBG
set casename=B1-AceVisc-Dp1000
set dirout=%name%_out

REM "executables" are renamed and called from their directory

set dirbin=../../bin/windows
set partvtk="%dirbin%/PartVTK4_win64.exe"
set createVtk="%dirbin%/pyvtker_v16.py"
set python3 = "python.exe"

if not "%ERRORLEVEL%" == "0" goto fail
set diroutdata=%dirout%\data
mkdir %diroutdata%

REM CODES are executed according the selected parameters of execution in this testcase
if exist %dirout%\particles\%casename%_*.vtk del %dirout%\particles\%casename%_*.vtk 

REM Executes PartVTK4 to create VTK files with particles.
set dirout2=%dirout%\particles
mkdir %dirout2%
rem no space between variables name
%partvtk% -dirin %diroutdata% -savecsv %dirout2%\%casename% -vars:Mass,Press,Qfxx,Qfxy,Qfxz,Qfyy,Qfyz,Qfzz,VonMises3D,GradVel,CellOffSpring,StrainDot,Acec,AceVisc
python %createVtk% %casename% %dirout%\particles
if not "%ERRORLEVEL%" == "0" goto fail


:success
echo All done
goto end

:fail
echo Execution aborted.

:end
pause
