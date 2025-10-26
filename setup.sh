#!/bin/bash
# Linux/macOS Setup Script
# This is a wrapper around setup.py for Unix-based systems

echo "======================================"
echo "Drone Detection System - Quick Setup"
echo "======================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ using your package manager"
    exit 1
fi

# Check Python version
python3 --version

# Run the setup script
python3 setup.py

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

