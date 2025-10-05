@echo off

REM Windows installer script for OmniBAR-POW
if not exist ".venv" (
    python -m venv .venv
)
call .venv\Scripts\activate.bat
pip install -r omnibar\requirements.txt
pip install -e .
pip install -r examples\requirements-workbench.txt
cd examples
python start_workbench.py