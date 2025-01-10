@echo off
cd /d "%~dp0"
cd ..
set TempFiles=%cd%\PyomoTempFiles
@echo on 
cd %TempFiles%
del *.lp
del *.log
del *.script
del *.sol
del *.dat
del *.txt

