@echo off
rem Builds and runs the hatching reproduction (tools/offline/HatchRepro.cpp). Output: build\release\hatch\
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "VCPKG_ROOT=C:\Dev\vcpkg"
cd /d "%~dp0"
cmake --build build\release --target HatchRepro || exit /b 1
if not exist build\release\hatch mkdir build\release\hatch
build\release\HatchRepro.exe build\release\hatch %*
exit /b %errorlevel%
