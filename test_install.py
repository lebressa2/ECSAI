#!/usr/bin/env python3
"""
Test script to verify the package can be built and installed
"""

import os
import sys
import subprocess

def test_build():
    """Test building the package"""
    print("Testing package build...")

    # Change to project directory
    project_dir = r"C:\dev\projects\ECSAIRefatored"
    os.chdir(project_dir)

    try:
        # Build the package
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", "."
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Package installed successfully!")
            print(result.stdout)

            # Test import
            test_import()
        else:
            print("❌ Installation failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)

    except Exception as e:
        print(f"❌ Error during build: {e}")

def test_import():
    """Test importing the installed package"""
    print("Testing package import...")

    try:
        import ecsaai
        print("✅ Package imported successfully!")
        print(f"Version: {getattr(ecsaai, '__version__', 'Not found')}")
        print(f"Available classes: {[x for x in dir(ecsaai) if not x.startswith('_')]}")

        # Test basic functionality
        if hasattr(ecsaai, 'Agent'):
            print("✅ Agent class found!")
        else:
            print("❌ Agent class not found!")

    except ImportError as e:
        print(f"❌ Import failed: {e}")

def test_remote_install():
    """Test installing from git URL"""
    print("Testing remote installation...")

    try:
        # This would be the command to install from git
        # For now, we'll just simulate it
        remote_url = "git+https://github.com/lebressa2/ECSAI.git"

        result = subprocess.run([
            sys.executable, "-m", "pip", "install", remote_url
        ], capture_output=True, text=True)

        print("Remote install result:")
        print("STDOUT:", result.stdout[-200:])  # Last 200 chars
        print("STDERR:", result.stderr[-200:])

    except Exception as e:
        print(f"Error testing remote install: {e}")

if __name__ == "__main__":
    if "--remote" in sys.argv:
        test_remote_install()
    else:
        test_build()
