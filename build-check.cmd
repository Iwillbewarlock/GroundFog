@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "VCPKG_ROOT=C:\Dev\vcpkg"
cd /d "%~dp0"
cmake --build build\release --target FogShaderCheck || exit /b 1
build\release\FogShaderCheck.exe
