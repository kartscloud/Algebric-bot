#!/usr/bin/env python3
"""
Generic Web Log Viewer - Run any Python script and view its output in browser

Usage:
    python run_with_web_logs.py lauren_v9.py
    python run_with_web_logs.py lauren_v7_final.py
    python run_with_web_logs.py any_script.py

Then open: http://YOUR_IP:8080
"""

import sys
import subprocess
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, time as dt_time
from collections import deque
import os
import glob
import time

# Store terminal output (last 1000 lines)
output_buffer = deque(maxlen=1000)
process = None
PORT = 8080
LOG_FILE = None  # Will be set based on script name
last_cleanup_date = None  # Track when we last cleaned up

class LogHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress HTTP server logs

    def do_GET(self):
        html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Trading Bot - Live Logs</title>
    <meta http-equiv="refresh" content="3">
    <style>
        body {{
            background: #0a0e27;
            color: #00ff00;
            font-family: 'Courier New', monospace;
            padding: 20px;
            margin: 0;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        h1 {{
            color: #fff;
            margin: 0;
            font-size: 24px;
        }}
        .info {{
            color: #a0aec0;
            font-size: 14px;
            margin-top: 5px;
        }}
        .status {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 5px;
            font-size: 12px;
            margin-left: 10px;
        }}
        .running {{ background: #48bb78; color: white; }}
        .stopped {{ background: #f56565; color: white; }}
        pre {{
            background: #1a1f3a;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            white-space: pre-wrap;
            border: 1px solid #2d3748;
            line-height: 1.5;
            max-height: 70vh;
            overflow-y: auto;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Trading Bot Monitor
            <span class="status {status_class}">{status}</span>
        </h1>
        <div class="info">
            Script: {script} |
            Log file: {log_file} |
            Lines: {line_count} |
            Auto-refresh: 3s |
            Last update: {time}
        </div>
    </div>
    <pre>{logs}</pre>
</body>
</html>
"""
        try:
            # Read from log file (includes historical logs)
            logs = ""
            line_count = 0

            if LOG_FILE and os.path.exists(LOG_FILE):
                with open(LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()
                    # Filter out loading animations and get last 500 lines
                    filtered = [l.rstrip() for l in lines if 'Loading Market Data' not in l and l.strip()]
                    filtered = filtered[-500:]  # Last 500 lines
                    logs = '\n'.join(filtered)
                    line_count = len(filtered)
            else:
                # Fall back to output buffer if no log file
                logs = '\n'.join(output_buffer) if output_buffer else "Waiting for logs..."
                line_count = len(output_buffer)

            # Determine status
            if process and process.poll() is None:
                status = "RUNNING"
                status_class = "running"
            else:
                status = "STOPPED"
                status_class = "stopped"

            page = html.format(
                time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                logs=logs,
                line_count=line_count,
                script=sys.argv[1] if len(sys.argv) > 1 else "N/A",
                log_file=LOG_FILE or "N/A",
                status=status,
                status_class=status_class
            )

            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(page.encode('utf-8'))
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Error: {e}".encode())

def cleanup_log_file():
    """Clean up log file at end of day (after 4 PM ET / market close)"""
    global last_cleanup_date

    if not LOG_FILE or not os.path.exists(LOG_FILE):
        return

    now = datetime.now()
    today = now.date()

    # Only cleanup once per day, after 4 PM (market close)
    if last_cleanup_date == today:
        return

    # Check if it's after 4 PM (16:00)
    if now.hour >= 16:
        try:
            # Keep only today's logs (last 200 lines as backup)
            with open(LOG_FILE, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()

            # Keep only last 200 lines
            if len(lines) > 200:
                with open(LOG_FILE, 'w', encoding='utf-8') as f:
                    f.write(f"=== Log cleaned at {now.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
                    f.write(f"=== Kept last 200 lines ===\n\n")
                    f.writelines(lines[-200:])

                print(f"[LOG CLEANUP] Cleaned {LOG_FILE}, kept last 200 lines")

            last_cleanup_date = today
        except Exception as e:
            print(f"[LOG CLEANUP] Error: {e}")

def find_log_file(script_name):
    """Find the log file based on script name"""
    # Common patterns
    base = os.path.splitext(script_name)[0]  # e.g., "lauren_v9"

    possible_paths = [
        f"{base}_logs/lauren.log",
        f"{base}_logs/{base}.log",
        f"{base}_data/logs.log",
        f"logs/{base}.log",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    # Try to find any matching log directory
    for pattern in [f"{base}*logs*/*.log", f"*{base}*/*.log"]:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]

    return None

def run_script(script_path):
    """Run the Python script and capture output"""
    global process

    try:
        # Run the script with UTF-8 encoding
        process = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1
        )

        output_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting {script_path}...")
        output_buffer.append("="*60)

        # Read output line by line (for terminal display)
        for line in iter(process.stdout.readline, ''):
            if line:
                # Handle carriage returns (progress indicators)
                if '\r' in line:
                    line = line.split('\r')[-1]

                line = line.rstrip()
                if line and 'Loading Market Data' not in line:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    output_buffer.append(f"[{timestamp}] {line}")
                    # Also print to terminal
                    print(line)

        process.wait()
        output_buffer.append("="*60)
        output_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Process exited with code {process.returncode}")

    except Exception as e:
        output_buffer.append(f"ERROR: {e}")

def start_web_server():
    """Start the web server"""
    try:
        server = HTTPServer(('0.0.0.0', PORT), LogHandler)
        print(f"\n{'='*60}")
        print(f"WEB LOG VIEWER RUNNING")
        print(f"Open in browser: http://localhost:{PORT}")
        print(f"Or from network: http://YOUR_EC2_IP:{PORT}")
        print(f"Log file: {LOG_FILE}")
        print(f"Auto-cleanup: Daily after 4 PM (keeps last 200 lines)")
        print(f"{'='*60}\n")
        server.serve_forever()
    except Exception as e:
        print(f"Web server error: {e}")

def log_cleanup_scheduler():
    """Background thread to check for log cleanup every hour"""
    while True:
        time.sleep(3600)  # Check every hour
        cleanup_log_file()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_with_web_logs.py <script.py>")
        print("\nExamples:")
        print("  python run_with_web_logs.py lauren_v9.py")
        print("  python run_with_web_logs.py lauren_v7_final.py")
        sys.exit(1)

    script_path = sys.argv[1]

    if not os.path.exists(script_path):
        print(f"Error: Script '{script_path}' not found")
        sys.exit(1)

    # Find the log file for this script
    LOG_FILE = find_log_file(script_path)
    if LOG_FILE:
        print(f"Found log file: {LOG_FILE}")
    else:
        print(f"Warning: No log file found for {script_path}, using live output only")

    # Start web server in background thread
    web_thread = threading.Thread(target=start_web_server, daemon=True)
    web_thread.start()

    # Start log cleanup scheduler in background thread
    cleanup_thread = threading.Thread(target=log_cleanup_scheduler, daemon=True)
    cleanup_thread.start()

    # Run the script (this blocks)
    run_script(script_path)
