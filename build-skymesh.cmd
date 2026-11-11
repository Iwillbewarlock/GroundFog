@echo off
rem Builds the sky-mesh reproduction (tools/offline/SkyMeshRepro.cpp) once per shader version and runs each.
rem Usage: build-skymesh.cmd [pitch up in degrees]. Output: build\release\skymesh\<commit>\
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
cd /d "%~dp0"
for %%C in (37c508f 5ddb7f9 9724a3e) do (
  if not exist build\release\skymesh\%%C mkdir build\release\skymesh\%%C
  git show %%C:src/RaymarchShaders.h > build\release\skymesh\%%C\RaymarchShaders.h || exit /b 1
  cl /nologo /std:c++latest /EHsc /O2 /W3 /DNOMINMAX /Ibuild\release\skymesh\%%C /Isrc /Fobuild\release\skymesh\%%C\ /Febuild\release\skymesh\%%C\SkyMeshRepro.exe tools\offline\SkyMeshRepro.cpp src\WindSim.cpp d3d11.lib d3dcompiler.lib dxguid.lib >build\release\skymesh\%%C\build.log 2>&1 || (type build\release\skymesh\%%C\build.log & exit /b 1)
  echo ==== shaders of %%C
  build\release\skymesh\%%C\SkyMeshRepro.exe build\release\skymesh\%%C %* || exit /b 1
)
exit /b 0
