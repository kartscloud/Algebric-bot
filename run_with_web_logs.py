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
from datetime import datetime
from collections import deque
import os

# Store terminal output (last 1000 lines)
output_buffer = deque(maxlen=1000)
process = None
PORT = 8080

class LogHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress HTTP server logs

    def do_GET(self):
        html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Trading Bot - Live Output</title>
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
        }}
        .error {{ color: #fc8181; }}
        .warning {{ color: #f6e05e; }}
        .success {{ color: #68d391; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Trading Bot Monitor
            <span class="status {status_class}">{status}</span>
        </h1>
        <div class="info">
            Script: {script} |
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
            logs = '\n'.join(output_buffer) if output_buffer else "Waiting for output..."

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
                line_count=len(output_buffer),
                script=sys.argv[1] if len(sys.argv) > 1 else "N/A",
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
            errors='replace',  # Replace invalid characters instead of crashing
            bufsize=1
        )

        output_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting {script_path}...")
        output_buffer.append("="*60)

        # Read output line by line
        for line in iter(process.stdout.readline, ''):
            if line:
                # Handle carriage returns (progress indicators)
                if '\r' in line:
                    # Only keep the last part after the last carriage return
                    line = line.split('\r')[-1]

                # Skip empty lines and repetitive loading animations
                line = line.rstrip()
                if line and 'Loading Market Data' not in line:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    output_buffer.append(f"[{timestamp}] {line}")

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
        print(f"{'='*60}\n")
        server.serve_forever()
    except Exception as e:
        print(f"Web server error: {e}")

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

    # Start web server in background thread
    web_thread = threading.Thread(target=start_web_server, daemon=True)
    web_thread.start()

    # Run the script (this blocks)
    run_script(script_path)
