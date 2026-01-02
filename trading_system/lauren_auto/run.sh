#!/bin/bash
# LAUREN v7 AGGRO - Quick Start Script

echo "=========================================="
echo "  LAUREN v7 AGGRO - Full Send Mode"
echo "=========================================="

# Check if requirements are installed
python3 -c "import robin_stocks" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run LAUREN
python3 lauren_v7_aggro.py "$@"
