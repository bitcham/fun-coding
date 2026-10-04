@echo off
setlocal EnableDelayedExpansion

rem ----- locate VS2022 via vswhere -----
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo [ERROR] vswhere.exe not found. Install VS2022 BuildTools or Community.
    exit /b 1
)

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -prerelease -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set "VSPATH=%%i"
)
if not defined VSPATH (
    echo [ERROR] VS2022 with VC tools not found.
    exit /b 1
)

echo [info] VS install: %VSPATH%

rem set up vcvars (skip if already set)
if not defined VSCMD_VER (
    call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat"
    if errorlevel 1 (
        echo [ERROR] vcvars64.bat failed
        exit /b 1
    )
)

rem ----- compile -----
pushd "%~dp0"
if not exist build mkdir build

set "CFLAGS=/nologo /W3 /O2 /utf-8 /D_CRT_SECURE_NO_WARNINGS"

rem H=8 build (default)
cl %CFLAGS% /Fo:build\h8\ /Fe:build\minimal_example_h8.exe data.c rng.c ngram.c nlm.c main.c
if errorlevel 1 (
    echo [ERROR] H=8 build failed
    popd
    exit /b 1
)

rem H=6 build
cl %CFLAGS% /DHIDDEN=6 /Fo:build\h6\ /Fe:build\minimal_example_h6.exe data.c rng.c ngram.c nlm.c main.c
if errorlevel 1 (
    echo [ERROR] H=6 build failed
    popd
    exit /b 1
)

echo [ok] build\minimal_example_h8.exe  (HIDDEN=8)
echo [ok] build\minimal_example_h6.exe  (HIDDEN=6)
popd
endlocal
