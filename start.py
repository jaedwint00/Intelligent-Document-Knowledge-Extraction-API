#!/usr/bin/env python3
"""
Startup script for the Intelligent Document & Knowledge Extraction API
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python 3.11+ is available"""
    if sys.version_info < (3, 11):
        print("❌ Python 3.11+ is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def check_virtual_environment():
    """Check if virtual environment is activated"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment is active")
        return True
    else:
        print("⚠️  Virtual environment not detected")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)

def create_directories():
    """Create necessary directories"""
    directories = ["uploads", "data", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Directories created")

def setup_environment():
    """Setup environment file"""
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            import shutil
            shutil.copy(".env.example", ".env")
            print("✅ Environment file created from .env.example")
        else:
            print("⚠️  No .env.example found, please create .env manually")

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting the API server...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app.main:app", 
            "--reload", 
            "--host", "0.0.0.0", 
            "--port", "8000"
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

def main():
    """Main startup function"""
    print("🔧 Intelligent Document & Knowledge Extraction API")
    print("=" * 50)
    
    check_python_version()
    
    if not check_virtual_environment():
        print("💡 Consider using a virtual environment:")
        print("   python3.11 -m venv venv")
        print("   source venv/bin/activate")
        print()
    
    install_dependencies()
    create_directories()
    setup_environment()
    
    print("\n🌐 API will be available at:")
    print("   • Main API: http://localhost:8000")
    print("   • Interactive Docs: http://localhost:8000/docs")
    print("   • ReDoc: http://localhost:8000/redoc")
    print("   • Health Check: http://localhost:8000/health")
    print()
    
    start_server()

if __name__ == "__main__":
    main()
