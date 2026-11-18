@echo off
rem Renders the far-layer comparison film's fog frames (tools/offline/HatchRepro.cpp, film mode) for three versions:
rem old = the 0.2.0 shader (commit 9e635b4), off = this shader with the far layer off (0.2.1), new = this shader.
rem Usage: build-farvideo.cmd [frames, default 192]. Output: build\release\farvideo\video_<old|off|new>.u8, labels and tris.
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
set "FRAMES=%~1"
if "%FRAMES%"=="" set "FRAMES=192"
set "OUT=build\release\farvideo"
if not exist %OUT%\old mkdir %OUT%\old
if not exist %OUT%\new mkdir %OUT%\new
git show 9e635b4:src/RaymarchShaders.h > %OUT%\old\RaymarchShaders.h || exit /b 1
cl /nologo /std:c++latest /EHsc /O2 /W3 /DNOMINMAX /I%OUT%\old /Isrc /Fo%OUT%\old\ /Fe%OUT%\old\HatchRepro.exe tools\offline\HatchRepro.cpp src\WindSim.cpp d3d11.lib d3dcompiler.lib dxguid.lib >%OUT%\old\build.log 2>&1 || (type %OUT%\old\build.log & exit /b 1)
cl /nologo /std:c++latest /EHsc /O2 /W3 /DNOMINMAX /Isrc /Fo%OUT%\new\ /Fe%OUT%\new\HatchRepro.exe tools\offline\HatchRepro.cpp src\WindSim.cpp d3d11.lib d3dcompiler.lib dxguid.lib >%OUT%\new\build.log 2>&1 || (type %OUT%\new\build.log & exit /b 1)
%OUT%\old\HatchRepro.exe %OUT% video old %FRAMES% || exit /b 1
%OUT%\new\HatchRepro.exe %OUT% video off %FRAMES% || exit /b 1
%OUT%\new\HatchRepro.exe %OUT% video new %FRAMES% || exit /b 1
exit /b 0
