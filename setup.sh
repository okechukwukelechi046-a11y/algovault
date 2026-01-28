#!/bin/bash

# AlgoVault Setup Script
echo "Setting up AlgoVault..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required. Please install Python 3.9 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "Python version: $PYTHON_VERSION"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up Python path
echo "Setting up Python path..."
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "export PYTHONPATH=$PYTHONPATH:$(pwd)" >> venv/bin/activate

# Run tests
echo "Running tests..."
pytest tests/ -v

# Generate documentation
echo "Generating documentation..."
cd docs && make html && cd ..

echo ""
echo "========================================="
echo "AlgoVault setup complete!"
echo "========================================="
echo ""
echo "Available commands:"
echo "  source venv/bin/activate      # Activate virtual environment"
echo "  pytest tests/ -v              # Run all tests"
echo "  python benchmarks/run_all_benchmarks.py  # Run benchmarks"
echo "  python visualizations/dijkstra_visualizer.py  # Run visualization"
echo "  jupyter notebook              # Start Jupyter notebook"
echo ""
echo "Documentation: file://$(pwd)/docs/_build/html/index.html"
echo "========================================="
