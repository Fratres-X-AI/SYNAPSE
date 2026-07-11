@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "LOG=%LOCALAPPDATA%\Synapse\launch.log"
if not exist "%LOCALAPPDATA%\Synapse" mkdir "%LOCALAPPDATA%\Synapse" >nul 2>&1

REM --- Source install: prefer Python when launcher script is present ---
if exist "synapse_launcher.py" (
    set "PY="
    set "PYW="

    if exist ".venv\Scripts\python.exe" (
        set "PY=.venv\Scripts\python.exe"
        if exist ".venv\Scripts\pythonw.exe" set "PYW=.venv\Scripts\pythonw.exe"
        goto :deps
    )

    where py >nul 2>&1
    if %errorlevel%==0 (
        py -3.11 -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3,11),(3,12)) else 1)" >nul 2>&1
        if %errorlevel%==0 (
            set "PY=py -3.11"
            set "PYW=pyw -3.11"
            goto :deps
        )
        py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2] in ((3,11),(3,12)) else 1)" >nul 2>&1
        if %errorlevel%==0 (
            set "PY=py -3.12"
            set "PYW=pyw -3.12"
            goto :deps
        )
    )

    where python >nul 2>&1
    if %errorlevel%==0 (
        set "PY=python"
        where pythonw >nul 2>&1
        if %errorlevel%==0 set "PYW=pythonw"
        goto :deps
    )
)

REM --- Frozen build (no Python required) ---
if exist "Synapse.exe" (
    start "" "%~dp0Synapse.exe" home
    exit /b 0
)
if exist "dist\Synapse.exe" (
    start "" "%~dp0dist\Synapse.exe" home
    exit /b 0
)

call :fail "Synapse could not find Python 3.11 or 3.12." ^
    "Install Python 3.11 from https://www.python.org/downloads/ (check Add to PATH)," ^
    "then run:  scripts\setup_windows.ps1" ^
    "Or download Synapse.exe from GitHub Releases."

:deps
%PY% -c "import mediapipe, cv2" >nul 2>&1
if %errorlevel% neq 0 (
    call :fail "Synapse dependencies are not installed yet." ^
        "From this folder, run once:  scripts\setup_windows.ps1" ^
        "Or manually:  py -3.11 -m venv .venv  &&  .venv\Scripts\pip install -r requirements.txt"
)

:launch_python
if defined PYW (
    start "" %PYW% "%~dp0synapse_launcher.py" home
) else (
    start "" %PY% "%~dp0synapse_launcher.py" home
)
exit /b 0

:fail
echo.
echo  Synapse launch failed
echo  ----------------------
shift
:fail_loop
if "%~1"=="" goto :fail_done
echo  %~1
shift
goto :fail_loop
:fail_done
echo.
echo  Log: %LOG%
echo.
pause
exit /b 1
