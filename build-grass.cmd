@echo off
rem Builds HatchRepro and runs its grass report (plan r12 step 0a): far grass under the fog with projection jitter,
rem now / exact march (DebugView 2, 64 steps) / exact24 (DebugView 3, 24 steps) / no fog, 28 scenes x 4 fog cases.
rem About 12 minutes on WARP (697 s; 285 s before exact24 was added), so not part of build.cmd.
rem Output: build\release\grass\grass.txt. Extra arguments go to HatchRepro (quick = one case, dump = frame 0 images,
rem thin 600 = also ThinFogStart 600 against 0 on a second line per case, only clumps = the scenes whose name holds it).
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "VCPKG_ROOT=C:\Dev\vcpkg"
cd /d "%~dp0"
cmake --build build\release --target HatchRepro || exit /b 1
if not exist build\release\grass mkdir build\release\grass
build\release\HatchRepro.exe build\release\grass grass %* > build\release\grass\grass.txt
set "RC=%errorlevel%"
type build\release\grass\grass.txt
exit /b %RC%
