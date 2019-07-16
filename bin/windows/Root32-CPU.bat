@echo off
REM "name" and "dirout" are named according to the testcase

set python3="C:/Users/AL43569/AppData/Local/Programs/Python/Python37/python.exe"
set name=Stu-Aniso
set casename=A1-RunAniso2d
set dirout=%name%_out

REM "executables" are renamed and called from their directory

set dirbin=../../../bin/windows
set gencase="%dirbin%/GenCase4_win64.exe"
rem set dualsphysicscpu="%dirbin%/DualSPHysics4.2_win64_NoNsph.exe"
set dualsphysicscpu="%dirbin%/DualSPHysics4.2_win64_debug.exe"
set dualsphysicsgpu="%dirbin%/DualSPHysics4.2_win64.exe"
set boundaryvtk="%dirbin%/BoundaryVTK4_win64.exe"
set partvtk="%dirbin%/PartVTK4_win64.exe"
set partvtkout="%dirbin%/PartVTKOut4_win64.exe"
set measuretool="%dirbin%/MeasureTool4_win64.exe"
set computeforces="%dirbin%/ComputeForces4_win64.exe"
set isosurface="%dirbin%/IsoSurface4_win64.exe"
set measureboxes="%dirbin%/MeasureBoxes4_win64.exe"
rem createVtk takes ths csv files and convert them in Ascii Vtk files
rem needs python3 and numpy to be launched
set createVtk="%dirbin%/convertCsvVtk.py"

REM "dirout" is created to store results or it is removed if it already exists

if not exist %dirout% mkdir %dirout%
if exist %dirout%\data\Part_*.bi4 del %dirout%\data\Part_*.bi4
if exist %dirout%\particles\%casename%_*.vtk del %dirout%\particles\%casename%_*.vtk

if not "%ERRORLEVEL%" == "0" goto fail
set diroutdata=%dirout%\data
mkdir %diroutdata%

REM CODES are executed according the selected parameters of execution in this testcase

REM Executes GenCase4 to create initial files for simulation.
%gencase% Def %dirout%/%name% -save:all
if not "%ERRORLEVEL%" == "0" goto fail

REM Executes DualSPHysics to simulate SPH method.
REM %dualsphysicscpu% -cpu %dirout%/%name% %dirout% -dirdataout data -svres
%dualsphysicscpu% -cpu %dirout%/%name% %dirout% -dirdataout data -svres -sv:binx
if not "%ERRORLEVEL%" == "0" goto fail

REM Executes PartVTK4 to create VTK files with particles.
set dirout2=%dirout%\particles
mkdir %dirout2%
rem no space between variables name
rem %partvtk% -dirin %diroutdata% -savevtk %dirout2%/%casename%All -vars:Mass,Press
%partvtk% -dirin %diroutdata% -savecsv %dirout2%/%casename%All -vars:Mass,Press,Qfxx,Qfyy,Qfzz,Qfyz,Qfxz,Qfxy
%python3% %createVtk% %casename% %dirout%/particles
if not "%ERRORLEVEL%" == "0" goto fail

REM Executes PartVTKOut4 to create VTK files with excluded particles.
REM %partvtkout% -dirin %diroutdata% -filexml %dirout%/%name%.xml -savevtk %dirout2%/PartFluidOut -SaveResume %dirout%/ResumeFluidOut
if not "%ERRORLEVEL%" == "0" goto fail

REM Executes MeasureTool4 to create VTK files with velocity and a CSV file with velocity at each simulation time.
set dirout2=%dirout%\velocity
REM mkdir %dirout2%
REM %measuretool% -dirin %diroutdata% -points CaseDambreak_PointsVelocity.txt -onlytype:-all,+fluid -vars:-all,+vel.x,+vel.m -savevtk %dirout2%/PointsVelocity -savecsv %dirout%/PointsVelocity
if not "%ERRORLEVEL%" == "0" goto fail

REM Executes MeasureTool4 to create VTK files with incorrect pressure and a CSV file with value at each simulation time.
REM set dirout2=%dirout%\pressure
REM mkdir %dirout2%
REM %measuretool% -dirin %diroutdata% -points CaseDambreak_PointsPressure_Incorrect.txt -onlytype:-all,+fluid -vars:-all,+press,+kcorr -kcusedummy:0 -kclimit:0.5 -savevtk %dirout2%/PointsPressure_Incorrect -savecsv %dirout%/PointsPressure_Incorrect
if not "%ERRORLEVEL%" == "0" goto fail

REM Executes MeasureTool4 to create VTK files with correct pressure and a CSV file with value at each simulation time.
REM %measuretool% -dirin %diroutdata% -points CaseDambreak_PointsPressure_Correct.txt -onlytype:-all,+fluid -vars:-all,+press,+kcorr -kcusedummy:0 -kclimit:0.5 -savevtk %dirout2%/PointsPressure_Correct -savecsv %dirout%/PointsPressure_Correct
if not "%ERRORLEVEL%" == "0" goto fail

REM Executes ComputeForces to create a CSV file with force at each simulation time.
REM %computeforces% -dirin %diroutdata% -filexml %dirout%/%name%.xml -onlymk:21 -savecsv %dirout%/WallForce
if not "%ERRORLEVEL%" == "0" goto fail

REM Executes IsoSurface4 to create VTK files with surface fluid and slices of surface.
REM set dirout2=%dirout%\surface
REM mkdir %dirout2%
REM set planesy="-slicevec:0:0.1:0:0:1:0 -slicevec:0:0.2:0:0:1:0 -slicevec:0:0.3:0:0:1:0 -slicevec:0:0.4:0:0:1:0 -slicevec:0:0.5:0:0:1:0 -slicevec:0:0.6:0:0:1:0"
REM set planesx="-slicevec:0.1:0:0:1:0:0 -slicevec:0.2:0:0:1:0:0 -slicevec:0.3:0:0:1:0:0 -slicevec:0.4:0:0:1:0:0 -slicevec:0.5:0:0:1:0:0 -slicevec:0.6:0:0:1:0:0 -slicevec:0.7:0:0:1:0:0 -slicevec:0.8:0:0:1:0:0 -slicevec:0.9:0:0:1:0:0 -slicevec:1.0:0:0:1:0:0"
REM set planesd="-slice3pt:0:0:0:1:0.7:0:1:0.7:1"
REM %isosurface% -dirin %diroutdata% -saveiso %dirout2%/Surface -vars:-all,vel,rhop,idp,type -saveslice %dirout2%/Slices %planesy% %planesx% %planesd%
if not "%ERRORLEVEL%" == "0" goto fail

REM Executes MeasureBoxes4 to create VTK files with particles assigned to different zones and a CSV file with information of each zone.
REM set dirout2=%dirout%\meaboxes
REM mkdir %dirout2%
REM %measureboxes% -dirin %diroutdata% -fileboxes CaseDambreak_FileBoxes.txt -savecsv %dirout%/ResultBoxes.csv -savevtk %dirout2%/Boxes.vtk
if not "%ERRORLEVEL%" == "0" goto fail

:success
echo All done
goto end

:fail
echo Execution aborted.

:end
pause
