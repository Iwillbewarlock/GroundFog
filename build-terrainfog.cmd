@echo off
rem The far layer on real land (tools/offline/TerrainFogRepro.cpp): builds it against the 0.2.0 shader (old) and this
rem one, and renders one view three ways: old, off (0.2.1) and new.
rem Usage: build-terrainfog.cmd <view name> <player x> <player y> <yaw> <pitch> <eye above ground>
rem Land: build\terrain\tamriel_box.f32 (tools\offline\land_from_housecarl.py; 545 x 673 from -24576, -69632).
rem Output: build\release\terrainfog\<view name>\
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
if "%~6"=="" (echo usage: build-terrainfog.cmd ^<view name^> ^<player x^> ^<player y^> ^<yaw^> ^<pitch^> ^<eye above ground^> & exit /b 2)
set "OUT=build\release\terrainfog"
if not exist %OUT%\old mkdir %OUT%\old
if not exist %OUT%\new mkdir %OUT%\new
if not exist %OUT%\%1 mkdir %OUT%\%1
git show 9e635b4:src/RaymarchShaders.h > %OUT%\old\RaymarchShaders.h || exit /b 1
cl /nologo /std:c++latest /EHsc /O2 /W3 /DNOMINMAX /I%OUT%\old /Isrc /Fo%OUT%\old\ /Fe%OUT%\old\TerrainFogRepro.exe tools\offline\TerrainFogRepro.cpp src\WindSim.cpp d3d11.lib d3dcompiler.lib dxguid.lib >%OUT%\old\build.log 2>&1 || (type %OUT%\old\build.log & exit /b 1)
cl /nologo /std:c++latest /EHsc /O2 /W3 /DNOMINMAX /Isrc /Fo%OUT%\new\ /Fe%OUT%\new\TerrainFogRepro.exe tools\offline\TerrainFogRepro.cpp src\WindSim.cpp d3d11.lib d3dcompiler.lib dxguid.lib >%OUT%\new\build.log 2>&1 || (type %OUT%\new\build.log & exit /b 1)
set "LAND=build\terrain\tamriel_box.f32 545 673 -24576 -69632"
%OUT%\old\TerrainFogRepro.exe %OUT%\%1 %LAND% %2 %3 %4 %5 %6 old || exit /b 1
%OUT%\new\TerrainFogRepro.exe %OUT%\%1 %LAND% %2 %3 %4 %5 %6 off || exit /b 1
%OUT%\new\TerrainFogRepro.exe %OUT%\%1 %LAND% %2 %3 %4 %5 %6 new 1 || exit /b 1
exit /b 0
