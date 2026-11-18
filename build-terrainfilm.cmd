@echo off
rem Film of the far layer on real land: the same builds as build-terrainfog.cmd (old = the 0.2.0 shader, new = this one),
rem one view rendered <frames> times as old, off (0.2.1) and new with the patches travelling.
rem Usage: build-terrainfilm.cmd <view name> <player x> <player y> <yaw> <pitch> <eye above ground> <frames>
rem Output: build\release\terrainfog\<view name>\video_<old|off|new>.u8 (tools\offline\terrain_film.py makes the mp4)
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
if "%~7"=="" (echo usage: build-terrainfilm.cmd ^<view name^> ^<player x^> ^<player y^> ^<yaw^> ^<pitch^> ^<eye above ground^> ^<frames^> & exit /b 2)
set "OUT=build\release\terrainfog"
if not exist %OUT%\old mkdir %OUT%\old
if not exist %OUT%\new mkdir %OUT%\new
if not exist %OUT%\%1 mkdir %OUT%\%1
git show 9e635b4:src/RaymarchShaders.h > %OUT%\old\RaymarchShaders.h || exit /b 1
cl /nologo /std:c++latest /EHsc /O2 /W3 /DNOMINMAX /I%OUT%\old /Isrc /Fo%OUT%\old\ /Fe%OUT%\old\TerrainFogRepro.exe tools\offline\TerrainFogRepro.cpp src\WindSim.cpp d3d11.lib d3dcompiler.lib dxguid.lib >%OUT%\old\build.log 2>&1 || (type %OUT%\old\build.log & exit /b 1)
cl /nologo /std:c++latest /EHsc /O2 /W3 /DNOMINMAX /Isrc /Fo%OUT%\new\ /Fe%OUT%\new\TerrainFogRepro.exe tools\offline\TerrainFogRepro.cpp src\WindSim.cpp d3d11.lib d3dcompiler.lib dxguid.lib >%OUT%\new\build.log 2>&1 || (type %OUT%\new\build.log & exit /b 1)
set "LAND=build\terrain\tamriel_box.f32 545 673 -24576 -69632"
%OUT%\old\TerrainFogRepro.exe %OUT%\%1 %LAND% %2 %3 %4 %5 %6 old 0 %7 || exit /b 1
%OUT%\new\TerrainFogRepro.exe %OUT%\%1 %LAND% %2 %3 %4 %5 %6 off 0 %7 || exit /b 1
%OUT%\new\TerrainFogRepro.exe %OUT%\%1 %LAND% %2 %3 %4 %5 %6 new 0 %7 || exit /b 1
exit /b 0
