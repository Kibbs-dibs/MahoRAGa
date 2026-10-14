@echo off
if exist "%~dp0.venv\Scripts\python.exe" (
    "%~dp0.venv\Scripts\python.exe" "%~dp0start.py" %*
) else (
    python "%~dp0start.py" %*
)
