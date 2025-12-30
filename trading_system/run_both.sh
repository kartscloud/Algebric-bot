#!/bin/bash
#
# Run both trading systems simultaneously
#

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         DUAL TRADING SYSTEM LAUNCHER                         ║"
echo "║                                                              ║"
echo "║  Starting:                                                   ║"
echo "║    • Quant Trader (50% buying power)                        ║"
echo "║    • LAUREN v6 (50% buying power)                           ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Start Quant Trader in background
echo "Starting Quant Trader..."
cd "$DIR/quant_trader"
python main.py &
QUANT_PID=$!
echo "  PID: $QUANT_PID"

# Wait a moment
sleep 2

# Start LAUREN in background
echo "Starting LAUREN..."
cd "$DIR/lauren_auto"
python lauren.py &
LAUREN_PID=$!
echo "  PID: $LAUREN_PID"

echo ""
echo "Both systems running!"
echo "Press Ctrl+C to stop both."
echo ""

# Handle Ctrl+C to kill both
trap "echo 'Stopping...'; kill $QUANT_PID $LAUREN_PID 2>/dev/null; exit" SIGINT SIGTERM

# Wait for both to finish (or Ctrl+C)
wait
