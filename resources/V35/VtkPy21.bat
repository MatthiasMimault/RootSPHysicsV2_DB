@echo off
REM "name" and "dirout" are named according to the testcase

set name=DBG
set casename=X1-MPGX-Dp500
set dirout=%name%_out

REM "executables" are renamed and called from their directory

set dirbin=../../bin/windows
set partvtk="%dirbin%/PartVTK4_win64.exe"
set createVtk="%dirbin%/pyvtker_v21.py"
set python3 = "python.exe"
set /A smooth = 10 
rem switch necessary to pass numerical values to argument

if not "%ERRORLEVEL%" == "0" goto fail
set diroutdata=%dirout%\data
mkdir %diroutdata%

REM CODES are executed according the selected parameters of execution in this testcase
if exist %dirout%\particles\%casename%_*.vtk del %dirout%\particles\%casename%_*.vtk 

REM Executes PartVTK4 to create VTK files with particles.
set dirout2=%dirout%\particles
mkdir %dirout2%
rem no space between variables name
rem %partvtk% -dirin %diroutdata% -savevtk %dirout2%\%casename%VtkOriginal -vars:Mass,Press,Qfxx,Qfxy,Qfxz,Qfyy,Qfyz,Qfzz,VonMises3D,GradVel,CellOffSpring,StrainDot,Acec
rem %partvtk% -dirin %diroutdata% -savecsv %dirout2%\%casename% -vars:Mass,Press,Qfxx,Qfxy,Qfxz,Qfyy,Qfyz,Qfzz,VonMises3D,GradVel,CellOffSpring,StrainDot,Acec,AceVisc
python %createVtk% %casename% %dirout%\particles %smooth%

if not "%ERRORLEVEL%" == "0" goto fail


:success
echo All done
goto end

:fail
echo Execution aborted.

:end
pause
