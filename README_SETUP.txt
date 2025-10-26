========================================
DRONE DETECTION SYSTEM - QUICK START
========================================

For new users who just cloned this repository:

AUTOMATIC SETUP (Recommended)
------------------------------

Windows:
  1. Double-click: setup.bat
  OR
  2. Run: python setup.py

Linux/macOS:
  1. Run: chmod +x setup.sh && ./setup.sh
  OR
  2. Run: python3 setup.py


AFTER SETUP
-----------

1. Activate virtual environment:
   
   Windows:
     .\drone_det_env\Scripts\Activate.ps1
   
   Linux/macOS:
     source drone_det_env/bin/activate

2. Configure your dataset in config/config.yaml

3. Prepare dataset:
   python main.py --mode prepare

4. Start training:
   python main.py --mode train


NEED HELP?
----------
See SETUP_GUIDE.md for detailed instructions.


IMPORTANT NOTES
---------------
- Always activate the virtual environment before running
- The virtual environment (drone_det_env/) is NOT in git
- Each user creates their own virtual environment
- This prevents path and dependency issues


========================================

