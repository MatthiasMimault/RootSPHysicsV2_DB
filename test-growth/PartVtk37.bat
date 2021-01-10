@echo off
REM "name" and "dirout" are named according to the testcase

set name=DBG
set casename=B1-Visible_growth
set dirout=%name%_out

REM "executables" are renamed and called from their directory

set dirbin=../bin/windows
set partvtk="%dirbin%/PartVTK4_win64.exe"
set createVtk="%dirbin%/pyvtker_v22.py"
set pyStats="%dirbin%/pystat_d17c.py"
set options="lsv"
set /A smooth = 1 

if not "%ERRORLEVEL%" == "0" goto fail
set diroutdata=%dirout%\data
mkdir %diroutdata%

REM CODES are executed according the selected parameters of execution in this testcase
if exist %dirout%\particles\%casename%_*.vtk del %dirout%\particles\%casename%_*.vtk 

REM Executes PartVTK4 to create VTK files with particles.
set dirout2=%dirout%\particles
set dirimg=%dirout%\img
mkdir %dirout2%
mkdir %dirimg%
rem no space between variables name
%partvtk% -dirin %diroutdata% -savecsv %dirout2%\%casename% -vars:Mass,Press,Qfxx,Qfxy,Qfxz,Qfyy,Qfyz,Qfzz,VonMises3D,GradVel,CellOffSpring,StrainDot,Acec,AceVisc
python %createVtk% %casename% %dirout%\particles %dirout%\vtk %smooth%
python %pyStats% %casename% %dirout2% %dirimg% %options% 1
if not "%ERRORLEVEL%" == "0" goto fail


:success
echo All done
goto end

:fail
echo Execution aborted.

:end
pause
