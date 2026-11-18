@echo off
rem Builds and runs the grass depth probe's offline check (tools/offline/GrassProbeCheck.cpp). Not part of build.cmd.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "VCPKG_ROOT=C:\Dev\vcpkg"
cd /d "%~dp0"
cmake --build build\release --target GrassProbeCheck || exit /b 1
build\release\GrassProbeCheck.exe
exit /b %errorlevel%
