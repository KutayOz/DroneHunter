"""
Automatic Setup Script for Drone Detection System
This script automatically creates a virtual environment and installs all dependencies.
Works on Windows, Linux, and macOS.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60 + "\n")


def print_success(text):
    """Print success message."""
    print(f"✓ {text}")


def print_error(text):
    """Print error message."""
    print(f"✗ {text}")


def print_info(text):
    """Print info message."""
    print(f"→ {text}")


def check_python_version():
    """Check if Python version is compatible."""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print_info(f"Python {version.major}.{version.minor}.{version.micro}")
    
    # Check for Python 3.12 specifically
    if version.major != 3 or version.minor != 12:
        print_error("Python 3.12 is required!")
        print_info("This project is tested and optimized for Python 3.12")
        print_info("Please install Python 3.12 from https://www.python.org/downloads/")
        print_info("For older Python versions, compatibility is not guaranteed")
        
        response = input("Continue anyway? (not recommended) [y/N]: ").strip().lower()
        if response != 'y':
            print_error("Setup cancelled. Please install Python 3.12")
            return False
    
    print_success("Python 3.12 detected - perfect!")
    return True


def get_venv_name():
    """Get the virtual environment name."""
    return "venv312"


def get_venv_path():
    """Get the path to the virtual environment."""
    return Path(get_venv_name())


def get_activation_command():
    """Get the activation command based on the OS."""
    venv_name = get_venv_name()
    
    if platform.system() == "Windows":
        return f".\\{venv_name}\\Scripts\\Activate.ps1"
    else:
        return f"source {venv_name}/bin/activate"


def create_virtual_environment():
    """Create a virtual environment."""
    print_header("Creating Virtual Environment")
    
    venv_path = get_venv_path()
    venv_name = get_venv_name()
    
    # Check if venv already exists
    if venv_path.exists():
        print_info(f"Virtual environment '{venv_name}' already exists")
        response = input("Do you want to recreate it? (y/N): ").strip().lower()
        
        if response == 'y':
            print_info(f"Removing old virtual environment...")
            import shutil
            shutil.rmtree(venv_path)
            print_success("Old environment removed")
        else:
            print_info("Using existing virtual environment")
            return True
    
    print_info(f"Creating virtual environment '{venv_name}' with Python 3.12...")
    
    try:
        # Ensure we're using Python 3.12
        if sys.version_info.minor != 12:
            print_error(f"Current Python version is {sys.version_info.major}.{sys.version_info.minor}")
            print_error("Virtual environment must be created with Python 3.12")
            print_info("Please run this script with Python 3.12: python3.12 setup.py")
            return False
        
        # Create venv with Python 3.12
        subprocess.run([sys.executable, "-m", "venv", venv_name], check=True)
        print_success(f"Virtual environment '{venv_name}' created successfully with Python 3.12")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to create virtual environment: {e}")
        return False


def get_python_executable():
    """Get the path to the Python executable in the virtual environment."""
    venv_name = get_venv_name()
    
    if platform.system() == "Windows":
        return Path(venv_name) / "Scripts" / "python.exe"
    else:
        return Path(venv_name) / "bin" / "python"


def install_dependencies():
    """Install required dependencies."""
    print_header("Installing Dependencies")
    
    python_exe = get_python_executable()
    
    if not python_exe.exists():
        print_error(f"Python executable not found: {python_exe}")
        return False
    
    # Upgrade pip first
    print_info("Upgrading pip...")
    try:
        subprocess.run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        print_success("pip upgraded")
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to upgrade pip: {e}")
    
    # Install requirements
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print_error("requirements.txt not found!")
        return False
    
    print_info("Installing packages from requirements.txt...")
    print_info("This may take several minutes...")
    
    try:
        # Run pip install with real-time output
        process = subprocess.Popen(
            [str(python_exe), "-m", "pip", "install", "-r", "requirements.txt"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(f"  {line.strip()}")
        
        process.wait()
        
        if process.returncode == 0:
            print_success("All dependencies installed successfully")
            return True
        else:
            print_error("Some packages failed to install")
            return False
            
    except Exception as e:
        print_error(f"Failed to install dependencies: {e}")
        return False


def create_data_yaml_template():
    """Create data.yaml if it doesn't exist."""
    print_header("Checking Configuration Files")
    
    if not Path("data.yaml").exists():
        print_info("data.yaml will be created when you run preparation")
        print_info("Run: python main.py --mode prepare")
    else:
        print_success("data.yaml exists")


def verify_installation():
    """Verify the installation."""
    print_header("Verifying Installation")
    
    python_exe = get_python_executable()
    
    # Test imports
    test_script = """
import sys
try:
    import torch
    import ultralytics
    import cv2
    import yaml
    print("SUCCESS: All required packages imported")
    sys.exit(0)
except ImportError as e:
    print(f"ERROR: Import failed - {e}")
    sys.exit(1)
"""
    
    try:
        result = subprocess.run(
            [str(python_exe), "-c", test_script],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print_success("All required packages verified")
            return True
        else:
            print_error("Package verification failed")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print_error("Verification timed out")
        return False
    except Exception as e:
        print_error(f"Verification failed: {e}")
        return False


def print_next_steps():
    """Print next steps for the user."""
    print_header("Setup Complete!")
    
    activation_cmd = get_activation_command()
    
    print("Next steps:\n")
    print(f"1. Activate the virtual environment:")
    print(f"   {activation_cmd}\n")
    print(f"2. Verify the setup (optional):")
    print(f"   python test_training_fix.py\n")
    print(f"3. Prepare your dataset:")
    print(f"   python main.py --mode prepare\n")
    print(f"4. Start training:")
    print(f"   python main.py --mode train\n")
    print(f"5. Or use the GUI:")
    print(f"   python main.py --mode gui\n")
    
    print("=" * 60)
    print("\nTip: Always activate the virtual environment before running the project!")
    print("=" * 60 + "\n")


def main():
    """Main setup routine."""
    print_header("Drone Detection System - Automatic Setup")
    print(f"Platform: {platform.system()}")
    print(f"Python: {sys.version}")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        print_error("\nSetup failed: Could not create virtual environment")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print_error("\nSetup completed with warnings")
        print_info("You may need to install some packages manually")
    
    # Verify installation
    verify_installation()
    
    # Check config files
    create_data_yaml_template()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
