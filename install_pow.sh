#!/bin/bash

# MacOS/Linux installer script for OmniBAR-POW
if [ ! -d ".venv" ]; then
    python3.10 -m venv .venv
fi
source .venv/bin/activate
pip install -r omnibar/requirements.txt
pip install -e .
pip install -r examples/requirements-workbench.txt
cd examples && python start_workbench.py