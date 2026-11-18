@echo off
setlocal EnableDelayedExpansion

set "ROOT=%~dp0"
set "BUILD=%ROOT%build\release"

set "VCVARS="
rem Visual Studio 18 (this PC) first, then 2022.
for %%V in (18 2022) do (
    for %%E in (BuildTools Community Professional Enterprise) do (
        for %%P in ("%ProgramFiles%" "%ProgramFiles(x86)%") do (
            if not defined VCVARS (
                if exist "%%~P\Microsoft Visual Studio\%%V\%%E\VC\Auxiliary\Build\vcvars64.bat" (
                    set "VCVARS=%%~P\Microsoft Visual Studio\%%V\%%E\VC\Auxiliary\Build\vcvars64.bat"
                )
            )
        )
    )
)

if not defined VCVARS (
    echo ERROR: no Visual Studio 18 or 2022 vcvars64.bat found.
    exit /b 1
)

if not defined VCPKG_ROOT set "VCPKG_ROOT=C:\Dev\vcpkg"

call "%VCVARS%" >nul
if errorlevel 1 (
    echo ERROR: vcvars64 failed.
    exit /b 1
)

cd /d "%ROOT%"

if not exist "%BUILD%\CMakeCache.txt" (
    cmake --preset release || exit /b 1
)

cmake --build "%BUILD%" || exit /b 1

if exist "%BUILD%\FogShaderCheck.exe" (
    "%BUILD%\FogShaderCheck.exe" > "%BUILD%\shader-check.log" 2>&1
    if errorlevel 1 (
        echo ERROR: the offline shader check failed:
        type "%BUILD%\shader-check.log"
        rem not exit /b here: from inside these two nested blocks it reached "cmd /c build.cmd" as exit code 0
        goto :fail
    )
    type "%BUILD%\shader-check.log"
)

echo Done.
endlocal
exit /b 0

:fail
exit /b 1
