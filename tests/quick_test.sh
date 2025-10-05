#!/bin/bash
# Quick Test Helper for OmniBAR
# 
# This script provides convenient shortcuts for common testing scenarios.
# Usage: ./quick_test.sh [scenario]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}🧪 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Ensure we're in the tests directory
if [[ ! -f "test_imports.py" ]]; then
    echo "❌ Please run this script from the tests/ directory"
    exit 1
fi

# Default scenario
SCENARIO=${1:-"fast"}

case $SCENARIO in
    "smoke"|"s")
        print_status "Running smoke test (imports only)"
        python run_tests.py imports
        ;;
    
    "fast"|"f")
        print_status "Running fast tests (no slow integrations)"
        python run_tests.py fast
        ;;
    
    "dev"|"development"|"d")
        print_status "Running development test suite"
        print_status "Step 1: Imports..."
        python run_tests.py imports
        print_success "Imports passed!"
        
        print_status "Step 2: Core objectives..."
        python run_tests.py objectives
        print_success "Objectives passed!"
        
        print_status "Step 3: Logging..."
        python run_tests.py logging
        print_success "All development tests passed! 🎉"
        ;;
    
    "comprehensive"|"full"|"all"|"c")
        print_status "Running comprehensive test suite with rich output"
        python test_all.py --fast
        ;;
    
    "complete")
        print_warning "Running COMPLETE test suite (including slow tests)"
        echo "This may take 5+ minutes and requires API keys..."
        read -p "Continue? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            python test_all.py
        else
            print_status "Cancelled."
            exit 0
        fi
        ;;
    
    "ci"|"pipeline")
        print_status "Running CI/CD pipeline test sequence"
        
        print_status "Stage 1: Fast validation..."
        if python run_tests.py fast; then
            print_success "Fast tests passed!"
        else
            print_error "Fast tests failed! Pipeline stopped."
            exit 1
        fi
        
        print_status "Stage 2: Core functionality..."
        if python run_tests.py core; then
            print_success "Core tests passed!"
        else
            print_error "Core tests failed! Pipeline stopped."
            exit 1
        fi
        
        print_success "CI/CD pipeline completed successfully! 🚀"
        ;;
    
    "pre-commit"|"pc")
        print_status "Running pre-commit test suite"
        python run_tests.py fast
        if [ $? -eq 0 ]; then
            print_success "All pre-commit tests passed! Safe to commit. 🎉"
        else
            print_error "Some tests failed. Please fix before committing."
            exit 1
        fi
        ;;
    
    "pre-push"|"pp")
        print_status "Running pre-push test suite"
        python test_all.py --fast
        if [ $? -eq 0 ]; then
            print_success "All pre-push tests passed! Safe to push. 🚀"
        else
            print_error "Some tests failed. Please fix before pushing."
            exit 1
        fi
        ;;
    
    "list"|"l")
        echo "📋 Available test scenarios:"
        echo
        echo "Quick Tests:"
        echo "  smoke, s           - Import tests only (~1s)"
        echo "  fast, f            - Fast tests only (~5s)"
        echo "  dev, d             - Development workflow (~10s)"
        echo
        echo "Comprehensive Tests:"
        echo "  comprehensive, c   - Full fast suite with rich output"
        echo "  complete           - Everything including slow tests (5+ min)"
        echo
        echo "Workflow Tests:"
        echo "  ci, pipeline       - CI/CD pipeline simulation"
        echo "  pre-commit, pc     - Pre-commit validation"
        echo "  pre-push, pp       - Pre-push validation"
        echo
        echo "Utilities:"
        echo "  list, l            - Show this help"
        echo "  status             - Show test environment status"
        echo
        echo "Examples:"
        echo "  ./quick_test.sh fast"
        echo "  ./quick_test.sh dev"
        echo "  ./quick_test.sh pre-commit"
        ;;
    
    "status")
        print_status "Checking test environment status..."
        echo
        
        # Check Python version
        python_version=$(python --version 2>&1)
        print_success "Python: $python_version"
        
        # Check if we're in conda environment
        if [[ ! -z "$CONDA_DEFAULT_ENV" ]]; then
            print_success "Conda environment: $CONDA_DEFAULT_ENV"
        else
            print_warning "Not in a conda environment"
        fi
        
        # Check .env file
        if [[ -f "../.env" ]]; then
            print_success ".env file exists"
            if grep -q "OPENAI_API_KEY" "../.env" 2>/dev/null; then
                print_success "OPENAI_API_KEY found in .env"
            else
                print_warning "OPENAI_API_KEY not found in .env"
            fi
        else
            print_warning ".env file not found (required for LLM tests)"
        fi
        
        # Check key dependencies
        deps=("rich" "dotenv" "langchain" "pydantic_ai")
        for dep in "${deps[@]}"; do
            if python -c "import $dep" 2>/dev/null; then
                print_success "$dep: Available"
            else
                print_warning "$dep: Not available (optional for some tests)"
            fi
        done
        
        echo
        print_status "Test file count:"
        echo "  Total test files: $(ls test_*.py | wc -l)"
        echo "  Available in runners: $(python run_tests.py --list | grep -c '\.py')"
        ;;
    
    *)
        print_error "Unknown scenario: $SCENARIO"
        echo
        echo "Use './quick_test.sh list' to see available scenarios"
        exit 1
        ;;
esac
