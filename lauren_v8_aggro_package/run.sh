#!/bin/bash

# LAUREN v8 AGGRO Runner

echo "=========================================="
echo "  LAUREN v8 AGGRO"
echo "  Realistic Fills + Aggressive Sizing"
echo "=========================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Check dependencies
python3 -c "import robin_stocks" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run
echo "Starting LAUREN v8 AGGRO..."
echo ""
python3 lauren_v8_aggro.py "$@"
