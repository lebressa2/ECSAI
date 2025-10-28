#!/usr/bin/env python3
"""
Check git status for ECSAI repository
"""

import subprocess
import os
import sys

def run_cmd(cmd, cwd=None):
    """Run a command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
        print(f"$ {cmd}")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print("-" * 40)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {cmd}: {e}")
        return False

# Change to ECSAI directory
project_dir = os.path.dirname(__file__)
print(f"Checking git status in: {project_dir}")

# Check git status
run_cmd("git status", cwd=project_dir)

# Check branches
run_cmd("git branch -a", cwd=project_dir)

# Check recent commits
run_cmd("git log --oneline -5", cwd=project_dir)

# Check if there are uncommitted changes
result = subprocess.run("git diff --name-only", shell=True, capture_output=True, text=True, cwd=project_dir)
if result.stdout.strip():
    print("Uncommitted changes:")
    print(result.stdout)
else:
    print("No uncommitted changes")
