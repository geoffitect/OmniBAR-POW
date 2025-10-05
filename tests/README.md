# OmniBAR Test Suite

Comprehensive tests organized by category with fast and complete test runners.

## Quick Start

```bash
cd tests/

# Fast tests (development)
python run_tests.py fast        # ~4s
python run_tests.py imports     # ~1s  

# Complete tests  
python test_all.py --fast       # Skip slow tests
python test_all.py              # Everything (~5min)
```

## Test Categories

| Category | Files | Speed | Description |
|----------|-------|-------|-------------|
| **imports** | 1 | ⚡ Fast | Package import verification |
| **core** | 1 | 🐌 Slow | Core benchmarker functionality |  
| **objectives** | 4 | ⚡ Mixed | All evaluation objectives |
| **integrations** | 3 | 🐌 Slow | LangChain, Pydantic AI, LLM Judge |
| **logging** | 5 | ⚡ Fast | Logging and auto-evaluators |
| **fast** | 8 | ⚡ Fast | All fast tests combined |

## Test Runners

### Simple Runner (`run_tests.py`)
Best for development and CI/CD.

```bash
python run_tests.py [category]  # Run specific category
python run_tests.py --list      # List categories
```

### Comprehensive Runner (`test_all.py`)  
Rich output with progress bars and detailed reporting.

```bash
python test_all.py --category core     # Specific category
python test_all.py --fast              # Skip slow tests
python test_all.py --verbose           # Detailed errors
python test_all.py --list              # Show all tests
```

## Setup

### Basic Setup
```bash
pip install -r omnibar/requirements.txt
```

### For LLM/Integration Tests  
Create `.env` in project root:
```bash
echo "OPENAI_API_KEY=your_key_here" >> .env
```

## Common Usage

```bash
# Development workflow
python run_tests.py imports     # Smoke test  
python run_tests.py fast        # Before commit
python test_all.py --fast       # Before PR

# Troubleshooting
python run_tests.py imports     # Check basic setup
python test_imports.py          # Run single test directly
```

## Test Symbols
- 🔑 **ENV** - Requires `.env` file with API keys
- 🌐 **NET** - Requires internet connection  
- 🐌 **SLOW** - Takes >30 seconds
