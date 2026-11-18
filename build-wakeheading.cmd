@echo off
rem Builds the trail-by-heading tool (tools/offline/WakeHeading.cpp) and runs it on a case file: build-wakeheading.cmd <cases.txt>
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "VCPKG_ROOT=C:\Dev\vcpkg"
cd /d "%~dp0"
cmake --build build\release --target WakeHeading || exit /b 1
if "%~1"=="" exit /b 0
build\release\WakeHeading.exe %*
exit /b %errorlevel%
