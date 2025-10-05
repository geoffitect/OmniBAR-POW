#!/usr/bin/env python3
"""
OmniBAR API Server
==================

Flask-based API server that provides real-time backend for the frontend POWorkbench.
Integrates with existing prompt refiner infrastructure and SQLite database.
"""

import sqlite3
import subprocess
import threading
import time
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
import webbrowser

# Import existing visualization components
from visualize_prompt_landscape import PromptLandscapeVisualizer


class OmniBarAPIServer:
    """Real-time API server for OmniBAR prompt optimization workbench."""

    def __init__(self, db_path: str = "prompt_refiner_results.db", port: int = 8080):
        """Initialize the API server."""
        self.db_path = db_path
        self.port = port
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for frontend requests

        # Configure upload settings
        self.app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
        self.upload_folder = Path("uploaded_pdfs")
        self.upload_folder.mkdir(exist_ok=True)

        # Initialize empty database if it doesn't exist
        self._initialize_database()

        # Track running processes
        self.running_processes: Dict[str, subprocess.Popen] = {}
        self.process_status: Dict[str, Dict] = {}

        # Setup routes
        self._setup_routes()

        print(f"🚀 OmniBAR API Server starting on port {port}")
        print(f"📊 Database: {db_path}")
        print(f"📄 PDF uploads: {self.upload_folder.absolute()}")

    def _initialize_database(self):
        """Initialize empty database with proper schema if it doesn't exist."""
        if os.path.exists(self.db_path):
            print(f"✅ Database already exists: {self.db_path}")
            return

        print(f"🔧 First-time setup: Creating empty database...")
        print(f"   Database path: {self.db_path}")

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create test_runs table
            cursor.execute("""
                CREATE TABLE test_runs (
                    run_timestamp TEXT PRIMARY KEY,
                    original_prompt TEXT NOT NULL,
                    exploration_depth INTEGER NOT NULL,
                    total_variations INTEGER NOT NULL,
                    test_document TEXT,
                    evaluation_type TEXT DEFAULT 'pydantic',
                    created_at TEXT NOT NULL
                )
            """)

            # Create prompt_variations table
            cursor.execute("""
                CREATE TABLE prompt_variations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_timestamp TEXT NOT NULL,
                    variation_id TEXT NOT NULL,
                    variation_prompt TEXT NOT NULL,
                    depth INTEGER NOT NULL,
                    changes TEXT,
                    completeness_score REAL NOT NULL,
                    type_score REAL NOT NULL,
                    format_score REAL NOT NULL,
                    pairing_score REAL NOT NULL,
                    overall REAL NOT NULL,
                    proteins TEXT,
                    organisms TEXT,
                    extracted_json TEXT,
                    tokens_input INTEGER DEFAULT 0,
                    tokens_output INTEGER DEFAULT 0,
                    efficiency_score REAL DEFAULT 0.0,
                    FOREIGN KEY (run_timestamp) REFERENCES test_runs (run_timestamp)
                )
            """)

            conn.commit()
            conn.close()

            print(f"✅ Empty database created successfully!")
            print("   Tables: test_runs, prompt_variations")
            print("🎉 Ready for your first prompt optimization run!")

        except Exception as e:
            print(f"❌ Error creating database: {e}")
            if os.path.exists(self.db_path):
                os.remove(self.db_path)  # Clean up partial file
            raise

    def _setup_routes(self):
        """Setup Flask routes for the API."""

        @self.app.route('/')
        def serve_frontend():
            """Serve the main frontend HTML."""
            return send_from_directory('.', 'frontend.html')

        @self.app.route('/api/runs')
        def get_runs():
            """Get list of all runs from database."""
            try:
                print(f"📊 Fetching runs from database: {self.db_path}")

                conn = sqlite3.connect(self.db_path, timeout=10.0)  # Add timeout
                cursor = conn.cursor()

                # Check if table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='test_runs'
                """)

                if not cursor.fetchone():
                    print("⚠️  Table 'test_runs' not found")
                    conn.close()
                    return jsonify([])  # Return empty list

                cursor.execute("""
                    SELECT run_timestamp, exploration_depth, total_variations,
                           created_at, original_prompt
                    FROM test_runs
                    ORDER BY created_at DESC
                """)

                runs = []
                for row in cursor.fetchall():
                    try:
                        runs.append({
                            'timestamp': row[0],
                            'depth': row[1],
                            'total_variations': row[2],
                            'created_at': row[3],
                            'original_prompt': (row[4][:100] + '...' if len(row[4]) > 100 else row[4]) if row[4] else 'No prompt'
                        })
                    except Exception as row_error:
                        print(f"⚠️  Error processing row: {row_error}")
                        continue

                conn.close()
                print(f"✅ Returning {len(runs)} runs")
                return jsonify(runs)

            except sqlite3.Error as db_error:
                print(f"❌ Database error in get_runs: {db_error}")
                return jsonify({'error': f'Database error: {str(db_error)}'}), 500
            except Exception as e:
                print(f"❌ Unexpected error in get_runs: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/runs/<timestamp>')
        def get_run_data(timestamp):
            """Get detailed data for a specific run."""
            try:
                print(f"📊 Fetching run data for: {timestamp}")

                # Validate timestamp format
                if not timestamp or len(timestamp) < 8:
                    print(f"❌ Invalid timestamp format: {timestamp}")
                    return jsonify({'error': 'Invalid timestamp format'}), 400

                # Use existing visualizer to get run data
                visualizer = PromptLandscapeVisualizer(self.db_path)

                try:
                    run_data = visualizer.get_run_data(timestamp)

                    # Validate run data
                    if not run_data:
                        print(f"❌ No data found for run: {timestamp}")
                        visualizer.close()
                        return jsonify({'error': f'Run {timestamp} not found'}), 404

                    # Transform for API response with safe defaults
                    api_data = {
                        'timestamp': run_data.get('run_timestamp', timestamp),
                        'original_prompt': run_data.get('original_prompt', 'No prompt'),
                        'exploration_depth': run_data.get('exploration_depth', 0),
                        'total_variations': run_data.get('total_variations', 0),
                        'created_at': run_data.get('created_at', ''),
                        'original_score': run_data.get('original_score', 0),
                        'variations': run_data.get('variations', []),
                        'best_score': max(v.get('overall', 0) for v in run_data.get('variations', [])) if run_data.get('variations') else 0,
                        'completed': len(run_data.get('variations', [])),
                        'segmented_data': visualizer.generate_score_segmented_data(run_data.get('variations', []))
                    }

                    visualizer.close()
                    print(f"✅ Returning data for run {timestamp}: {api_data['completed']} variations")
                    return jsonify(api_data)

                except Exception as viz_error:
                    print(f"❌ Visualizer error for run {timestamp}: {viz_error}")
                    visualizer.close()
                    return jsonify({'error': f'Error processing run data: {str(viz_error)}'}), 500

            except Exception as e:
                print(f"❌ Unexpected error in get_run_data({timestamp}): {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/start-run', methods=['POST'])
        def start_run():
            """Start a new prompt refinement run."""
            try:
                # Handle both JSON and form data (with file uploads)
                if request.content_type and 'multipart/form-data' in request.content_type:
                    # Handle file upload
                    prompt = request.form.get('prompt', '').strip()
                    depth = int(request.form.get('depth', 2))
                    timestamp = request.form.get('timestamp') or datetime.now().strftime('%Y%m%d_%H%M%S')
                    use_ollama = request.form.get('useOllama', 'false').lower() == 'true'
                    api_key = request.form.get('apiKey', '').strip()
                    ollama_model = request.form.get('ollamaModel', 'llama3.1:latest')

                    # Handle PDF file
                    if 'pdf' not in request.files:
                        return jsonify({'error': 'PDF file is required'}), 400

                    pdf_file = request.files['pdf']
                    if pdf_file.filename == '':
                        return jsonify({'error': 'No PDF file selected'}), 400

                    if not pdf_file.filename.lower().endswith('.pdf'):
                        return jsonify({'error': 'File must be a PDF'}), 400

                    # Save uploaded PDF
                    filename = secure_filename(f"{timestamp}_{pdf_file.filename}")
                    pdf_path = self.upload_folder / filename
                    pdf_file.save(str(pdf_path))

                    print(f"📄 Saved PDF: {pdf_path}")

                else:
                    # Handle JSON data (backward compatibility)
                    data = request.get_json()
                    prompt = data.get('prompt', '').strip()
                    depth = data.get('depth', 2)
                    timestamp = data.get('timestamp') or datetime.now().strftime('%Y%m%d_%H%M%S')
                    use_ollama = data.get('useOllama', False)
                    api_key = data.get('apiKey', '').strip()
                    ollama_model = data.get('ollamaModel', 'llama3.1:latest')
                    pdf_path = Path("test_proteins.pdf")  # Default fallback - test file in current directory

                if not prompt:
                    return jsonify({'error': 'Prompt is required'}), 400

                # Check if already running
                if timestamp in self.running_processes:
                    return jsonify({'error': 'Run already in progress'}), 409

                # Build command based on model configuration - use absolute paths
                # Use the same Python executable that's running this server
                python_exe = sys.executable
                cmd = [
                    python_exe, 'prompt_refiner_pydantic.py',
                    '--prompt', prompt,
                    '--depth', str(depth),
                    '--pdf', str(pdf_path.absolute())
                ]

                # Add model configuration
                if use_ollama:
                    cmd.extend(['--ollama', '--ollama-model', ollama_model])
                    model_info = f"Ollama {ollama_model}"
                else:
                    model_info = "OpenAI"
                    if api_key:
                        # Set API key as environment variable for the process
                        os.environ['OPENAI_API_KEY'] = api_key
                        model_info += " (custom key)"
                    else:
                        model_info += " (.env key)"

                # Debug: Check if files exist
                if not pdf_path.exists():
                    print(f"❌ PDF file not found: {pdf_path.absolute()}")
                    return jsonify({'error': f'PDF file not found: {pdf_path.name}'}), 400

                if not Path('prompt_refiner_pydantic.py').exists():
                    print("❌ Script not found: prompt_refiner_pydantic.py")
                    return jsonify({'error': 'prompt_refiner_pydantic.py not found in current directory'}), 500

                print(f"🔄 Starting run {timestamp} with {model_info}")
                print(f"📄 Processing PDF: {pdf_path.name} ({pdf_path.stat().st_size / 1024:.1f} KB)")
                print(f"📁 Working directory: {Path.cwd()}")
                print(f"🐍 Python executable: {python_exe}")
                print(f"   Command: {' '.join(cmd[:8])}...")  # Don't log full command for security

                # Set up environment
                env = os.environ.copy()
                if api_key and not use_ollama:
                    env['OPENAI_API_KEY'] = api_key

                # Debug environment info
                virtual_env = env.get('VIRTUAL_ENV', 'None')
                python_path = env.get('PATH', 'Not set')[:100] + '...'  # First 100 chars
                print(f"🔧 Virtual env: {virtual_env}")
                print(f"🛤️  PATH (first 100): {python_path}")

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    env=env
                )

                # Track the process
                self.running_processes[timestamp] = process
                self.process_status[timestamp] = {
                    'status': 'running',
                    'started_at': datetime.now().isoformat(),
                    'progress': 0,
                    'total': 0,
                    'best_score': 0.0,
                    'current_variation': 0,
                    'pdf_file': str(pdf_path.name),
                    'model_config': {
                        'use_ollama': use_ollama,
                        'model_name': ollama_model if use_ollama else 'gpt-4',
                        'model_info': model_info
                    }
                }

                # Start monitoring thread
                monitor_thread = threading.Thread(
                    target=self._monitor_process,
                    args=(timestamp, process),
                    daemon=True
                )
                monitor_thread.start()

                return jsonify({
                    'runId': timestamp,
                    'status': 'started',
                    'modelConfig': model_info,
                    'pdfFile': str(pdf_path.name)
                })

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/stop-run/<timestamp>', methods=['POST'])
        def stop_run(timestamp):
            """Stop a running prompt refinement process."""
            try:
                if timestamp not in self.running_processes:
                    return jsonify({'error': 'No such run found'}), 404

                process = self.running_processes[timestamp]

                # Terminate the process gracefully
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

                # Clean up
                del self.running_processes[timestamp]
                if timestamp in self.process_status:
                    self.process_status[timestamp]['status'] = 'stopped'
                    self.process_status[timestamp]['stopped_at'] = datetime.now().isoformat()

                print(f"🛑 Stopped run {timestamp}")

                return jsonify({
                    'runId': timestamp,
                    'status': 'stopped'
                })

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/status/<timestamp>')
        def get_run_status(timestamp):
            """Get real-time status of a running process."""
            try:
                if timestamp in self.process_status:
                    status = self.process_status[timestamp].copy()

                    # Check if process is still running
                    if timestamp in self.running_processes:
                        process = self.running_processes[timestamp]
                        if process.poll() is not None:
                            # Process has finished
                            status['status'] = 'completed'
                            status['finished_at'] = datetime.now().isoformat()
                            del self.running_processes[timestamp]

                    return jsonify(status)
                else:
                    return jsonify({'error': 'Run not found'}), 404

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/generate-viz/<timestamp>')
        def generate_visualization(timestamp):
            """Generate visualization for a specific run."""
            try:
                visualizer = PromptLandscapeVisualizer(self.db_path)
                output_file = visualizer.generate_html(timestamp)
                visualizer.close()

                return jsonify({
                    'status': 'generated',
                    'file': output_file,
                    'url': f'/viz/{output_file}'
                })

            except Exception as e:
                return jsonify({'error': str(e)}), 500

        @self.app.route('/viz/<filename>')
        def serve_visualization(filename):
            """Serve generated visualization files."""
            return send_from_directory('.', filename)

        @self.app.route('/api/health')
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'running_processes': len(self.running_processes),
                'database': os.path.exists(self.db_path)
            })

    def _monitor_process(self, timestamp: str, process: subprocess.Popen):
        """Monitor a running process and update status."""
        try:
            while process.poll() is None:
                # Read output to track progress
                if process.stdout:
                    line = process.stdout.readline()
                    if line:
                        self._parse_progress_line(timestamp, line.strip())

                time.sleep(0.5)

            # Process finished
            if timestamp in self.process_status:
                return_code = process.poll()
                if return_code == 0:
                    self.process_status[timestamp]['status'] = 'completed'
                    print(f"✅ Run {timestamp} completed successfully")
                else:
                    self.process_status[timestamp]['status'] = 'failed'
                    print(f"❌ Run {timestamp} failed with code {return_code}")

                    # Capture stderr for debugging
                    try:
                        stderr_output = process.stderr.read() if process.stderr else "No stderr available"
                        if stderr_output:
                            print(f"📝 Error details: {stderr_output[:500]}...")  # First 500 chars
                            self.process_status[timestamp]['error_details'] = stderr_output[:1000]
                    except Exception as e:
                        print(f"Could not read stderr: {e}")

                self.process_status[timestamp]['finished_at'] = datetime.now().isoformat()

            # Clean up
            if timestamp in self.running_processes:
                del self.running_processes[timestamp]

        except Exception as e:
            print(f"Error monitoring process {timestamp}: {e}")

    def _parse_progress_line(self, timestamp: str, line: str):
        """Parse progress information from process output."""
        try:
            if timestamp not in self.process_status:
                return

            status = self.process_status[timestamp]

            # Look for progress indicators in the output
            if "Generating variation" in line:
                # Extract variation number
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.isdigit():
                        status['current_variation'] = int(part)
                        break

            elif "Total variations to generate:" in line:
                # Extract total count
                parts = line.split()
                for part in parts:
                    if part.isdigit():
                        status['total'] = int(part)
                        break

            elif "Score:" in line and "overall" in line:
                # Extract score information
                try:
                    score_part = line.split("overall")[1].strip()
                    score = float(score_part.split()[0])
                    if score > status.get('best_score', 0):
                        status['best_score'] = score
                        print(f"📈 Run {timestamp}: new best score {score:.3f}")
                except TypeError:
                    pass

            # Update progress percentage
            if status.get('total', 0) > 0:
                status['progress'] = (status.get('current_variation', 0) / status['total']) * 100

        except Exception as e:
            print(f"Error parsing progress line: {e}")

    def run(self, debug: bool = False, open_browser: bool = True):
        """Start the API server."""
        if open_browser:
            # Open browser after a short delay
            def open_browser_delayed():
                time.sleep(1.5)
                webbrowser.open(f'http://localhost:{self.port}')

            browser_thread = threading.Thread(target=open_browser_delayed, daemon=True)
            browser_thread.start()

        print(f"🌐 Frontend available at: http://localhost:{self.port}")
        print(f"📡 API endpoints at: http://localhost:{self.port}/api/")
        print(f"🔧 Health check: http://localhost:{self.port}/api/health")
        print("\n🧬 OmniBAR Workbench is ready!")
        print("Press Ctrl+C to stop the server")

        try:
            self.app.run(host='0.0.0.0', port=self.port, debug=debug, threaded=True)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down OmniBAR API Server...")
            self._cleanup()

    def _cleanup(self):
        """Clean up running processes on shutdown."""
        for timestamp, process in list(self.running_processes.items()):
            try:
                print(f"🧹 Terminating run {timestamp}")
                process.terminate()
                process.wait(timeout=3)
            except NameError:
                process.kill()

        self.running_processes.clear()
        self.process_status.clear()


def main():
    """Main entry point for the API server."""
    import argparse

    parser = argparse.ArgumentParser(description='OmniBAR API Server')
    parser.add_argument('--db', type=str, default='prompt_refiner_results.db',
                       help='Path to SQLite database')
    parser.add_argument('--port', type=int, default=8080,
                       help='Port to serve on')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--no-browser', action='store_true',
                       help='Don\'t open browser automatically')

    args = parser.parse_args()

    # Start the server (database will be auto-initialized if needed)
    server = OmniBarAPIServer(db_path=args.db, port=args.port)
    server.run(debug=args.debug, open_browser=not args.no_browser)


if __name__ == '__main__':
    main()