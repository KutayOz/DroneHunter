#!/bin/bash
# Linux/macOS Setup Script
# This is a wrapper around setup.py for Unix-based systems

echo "======================================"
echo "Drone Detection System - Quick Setup"
echo "======================================"
echo

# Check if Python 3.12 is installed
if ! command -v python3.12 &> /dev/null; then
    echo "ERROR: Python 3.12 is not installed"
    echo "Trying with python3..."
    if ! command -v python3 &> /dev/null; then
        echo "ERROR: Python 3 is not installed"
        echo "Please install Python 3.12 from https://www.python.org/downloads/"
        exit 1
    fi
    PYTHON_CMD=python3
else
    PYTHON_CMD=python3.12
fi

# Check Python version
$PYTHON_CMD --version
echo

# Run the setup script
$PYTHON_CMD setup.py

if [ $? -ne 0 ]; then
    echo
    echo "Setup failed! Please check the error messages above."
    exit 1
fi

echo
echo "======================================"
echo "Setup completed successfully!"
echo "======================================"
echo

