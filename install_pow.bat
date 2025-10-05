@echo off

REM Windows installer script for OmniBAR-POW
echo 🧬 OmniBAR-POW Windows Installer
echo ================================

if not exist ".venv" (
    echo 🔧 Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
)

echo 🔌 Activating virtual environment...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)

echo 📦 Installing dependencies...
pip install -r omnibar\requirements.txt
pip install -e .
pip install -r examples\requirements-workbench.txt

echo 🚀 Starting OmniBAR workbench...
cd examples
..\\.venv\\Scripts\\python.exe start_workbench.py

pause