@echo off
rem Builds and runs the valley / pooled-fog reproduction (tools/offline/PoolBoxRepro.cpp). Output: build\release\poolbox\
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "VCPKG_ROOT=C:\Dev\vcpkg"
cd /d "%~dp0"
cmake --build build\release --target PoolBoxRepro || exit /b 1
if not exist build\release\poolbox mkdir build\release\poolbox
build\release\PoolBoxRepro.exe build\release\poolbox %*
exit /b %errorlevel%
