#!/usr/bin/env python3
"""
OmniBAR Workbench Launcher
==========================

One-click startup script for the complete OmniBAR prompt optimization workbench.
Automatically starts the API server and opens the frontend in your default browser.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def check_dependencies():
    """Check if all required dependencies are available."""
    required_files = [
        'frontend.html',
        'api_server.py',
        'prompt_refiner_pydantic.py',
        'visualize_prompt_landscape.py'
    ]

    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all OmniBAR files are in the current directory.")
        return False

    return True


def check_python_packages():
    """Check if required Python packages are installed."""
    required_packages = [
        'flask', 'flask_cors', 'sqlite3', 'plotly', 'pydantic'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required Python packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False

    return True


def find_available_port(start_port=8080, max_attempts=10):
    """Find an available port starting from start_port."""
    import socket

    for i in range(max_attempts):
        port = start_port + i
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('localhost', port))
            sock.close()
            return port
        except OSError:
            continue

    return None


def start_workbench(port=8080, debug=False, no_browser=False):
    """Start the complete OmniBAR workbench."""

    print("🧬 OmniBAR Prompt Optimization Workbench")
    print("=" * 50)

    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        return False

    if not check_python_packages():
        return False

    print("✅ All dependencies found")

    # Find available port
    available_port = find_available_port(port)
    if not available_port:
        print(f"❌ Could not find available port starting from {port}")
        return False

    if available_port != port:
        print(f"⚠️  Port {port} busy, using port {available_port}")

    # Check for existing database
    db_path = Path("prompt_refiner_results.db")
    if db_path.exists():
        print(f"📊 Found existing database: {db_path}")
    else:
        print("📊 First-time setup - database will be created automatically")

    # Start the API server
    print(f"\n🚀 Starting OmniBAR API server on port {available_port}...")

    cmd = [
        sys.executable, 'api_server.py',
        '--port', str(available_port)
    ]

    if debug:
        cmd.append('--debug')

    if no_browser:
        cmd.append('--no-browser')

    try:
        # Start the server
        _ = subprocess.run(cmd)
        return True

    except KeyboardInterrupt:
        print("\n🛑 Workbench stopped by user")
        return True

    except Exception as e:
        print(f"❌ Error starting workbench: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Start the OmniBAR Prompt Optimization Workbench',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_workbench.py                    # Start with default settings
  python start_workbench.py --port 9000       # Use custom port
  python start_workbench.py --debug           # Enable debug mode
  python start_workbench.py --no-browser      # Don't open browser

The workbench provides:
  🧬 Real-time prompt optimization
  📊 Live visualization with mutation analysis
  🔄 Interactive run management
  💾 Complete database browsing
  🚀 One-click optimization runs
        """
    )

    parser.add_argument(
        '--port', type=int, default=8080,
        help='Port to serve on (default: 8080)'
    )

    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug mode'
    )

    parser.add_argument(
        '--no-browser', action='store_true',
        help='Don\'t open browser automatically'
    )

    parser.add_argument(
        '--check', action='store_true',
        help='Just check dependencies and exit'
    )

    args = parser.parse_args()

    if args.check:
        print("🔍 Checking OmniBAR dependencies...")
        deps_ok = check_dependencies() and check_python_packages()
        if deps_ok:
            print("✅ All dependencies satisfied")
            return 0
        else:
            print("❌ Missing dependencies")
            return 1

    # Start the workbench
    success = start_workbench(
        port=args.port,
        debug=args.debug,
        no_browser=args.no_browser
    )

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())