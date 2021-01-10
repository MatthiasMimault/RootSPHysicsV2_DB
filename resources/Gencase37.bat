@echo off
REM "name" and "dirout" are named according to the testcase

set name=DBG
set dirout=%name%_out

REM "executables" are renamed and called from their directory

set dirbin=../bin/windows
set gencase="%dirbin%/GenCase4_win64.exe"

REM "dirout" is created to store results or it is removed if it already exists
if not exist %dirout% mkdir %dirout%
del %dirout%\*.vtk
del %dirout%\*.bi4
del %dirout%\*.xml
if exist %dirout%\data\Part_*.bi4 del %dirout%\data\Part_*.bi4

rem Should it be moved to PartVtk ?
rem if exist %dirout%\particles\%casename%_*.vtk del %dirout%\particles\%casename%_*.vtk 

if not "%ERRORLEVEL%" == "0" goto fail
set diroutdata=%dirout%\data
mkdir %diroutdata%

REM CODES are executed according the selected parameters of execution in this testcase

REM Executes GenCase4 to create initial files for simulation.
%gencase% Def %dirout%/%name% -save:all
if not "%ERRORLEVEL%" == "0" goto fail

:success
echo All done
goto end

:fail
echo Execution aborted.

:end
pause
