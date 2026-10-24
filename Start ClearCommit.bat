@echo off
rem ============================================================
rem  ClearCommit TaskForge - double-click to start.
rem  Native desktop window over the live interpreter, with
rem  Conversation Clarity. First run installs dependencies.
rem ============================================================
setlocal
cd /d "%~dp0"
title ClearCommit TaskForge

where python >nul 2>nul
if errorlevel 1 (
  echo.
  echo  Python was not found on PATH.
  echo  Install Python 3.12+ from https://www.python.org/downloads/
  echo  and re-run this launcher.
  echo.
  pause
  exit /b 1
)

if not exist "app.py" (
  echo  app.py not found next to this launcher.
  pause
  exit /b 1
)

python -c "import webview" >nul 2>nul
if errorlevel 1 (
  echo  Installing dependencies ^(first run only^)...
  python -m pip install -q -r requirements.txt
)

python app.py %*
if errorlevel 1 (
  echo.
  echo  ClearCommit exited with an error. See the console output above.
  pause
)
endlocal
