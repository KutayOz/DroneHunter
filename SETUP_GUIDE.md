# Setup Guide for New Users

This guide explains how to set up the Drone Detection System after cloning from Git.

## 🚀 Quick Start (Automatic Setup)

### For Windows Users

**Option 1: Double-click the batch file**
```
Double-click: setup.bat
```

**Option 2: Run from command line**
```powershell
python setup.py
```

### For Linux/macOS Users

**Option 1: Run the shell script**
```bash
chmod +x setup.sh
./setup.sh
```

**Option 2: Run with Python**
```bash
python3 setup.py
```

---

## What the Setup Script Does

The automatic setup script (`setup.py`) will:

1. ✅ Check your Python version (requires 3.8+)
2. ✅ Create a virtual environment (`drone_det_env`)
3. ✅ Upgrade pip to the latest version
4. ✅ Install all required dependencies from `requirements.txt`
5. ✅ Verify the installation
6. ✅ Show you the next steps

**Time required**: ~5-10 minutes (depending on internet speed)

---

## 📋 Prerequisites

Before running setup, make sure you have:

- **Python 3.8 or higher** installed
  - Windows: Download from [python.org](https://www.python.org/)
  - Linux: `sudo apt install python3 python3-pip python3-venv`
  - macOS: `brew install python3`

- **Git** (to clone the repository)
  - Download from [git-scm.com](https://git-scm.com/)

- **NVIDIA GPU with CUDA support** (optional, for faster training)
  - CPU training also works, just slower

---

## 🔧 Manual Setup (If Automatic Setup Fails)

If the automatic setup doesn't work, follow these steps:

### Step 1: Clone the Repository
```bash
git clone <your-repo-url>
cd <repo-directory>
```

### Step 2: Create Virtual Environment

**Windows:**
```powershell
python -m venv drone_det_env
.\drone_det_env\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
python3 -m venv drone_det_env
source drone_det_env/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import torch; import ultralytics; print('All packages installed successfully!')"
```

---

## 🎯 After Setup

### 1. Activate Virtual Environment

You need to activate the virtual environment **every time** you work on the project:

**Windows:**
```powershell
.\drone_det_env\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
source drone_det_env/bin/activate
```

You'll see `(drone_det_env)` in your terminal prompt when activated.

### 2. Configure Your Dataset

Edit `config/config.yaml` and set your dataset path:
```yaml
dataset:
  primary_dataset: "Datasets/your-dataset-folder"
```

### 3. Prepare Dataset
```bash
python main.py --mode prepare
```

This will:
- Validate your dataset structure
- Create `data.yaml` with correct paths
- Check for any issues

### 4. Start Training
```bash
python main.py --mode train
```

Or use the GUI:
```bash
python main.py --mode gui
```

---

## 📁 Project Structure After Setup

```
drone-detection/
├── drone_det_env/          # Virtual environment (NOT in git)
├── config/
│   └── config.yaml         # Configuration file
├── src/
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── inference.py
│   └── gui.py
├── Datasets/               # Your datasets go here
├── runs/                   # Training outputs
├── logs/                   # Log files
├── models/                 # Saved models
├── main.py                 # Main entry point
├── setup.py               # Automatic setup script
├── setup.bat              # Windows setup wrapper
├── setup.sh               # Linux/macOS setup wrapper
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore file
└── README.md             # Project documentation
```

---

## ❓ Common Issues

### Issue: "python: command not found"
**Solution**: Install Python 3.8+ from python.org

### Issue: "Permission denied" on Linux/macOS
**Solution**: 
```bash
chmod +x setup.sh
```

### Issue: Virtual environment not activating on Windows
**Solution**: Enable script execution:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: CUDA out of memory during training
**Solution**: Reduce batch size in `config/config.yaml`:
```yaml
model:
  batch_size: 4  # or even 2
```

### Issue: NumPy MINGW warning
**Solution**: Reinstall NumPy:
```bash
pip uninstall numpy -y
pip install numpy --force-reinstall --no-cache-dir
```

---

## 🔄 Updating the Project

When pulling new changes from git:

```bash
# Pull latest changes
git pull origin main

# Activate virtual environment
.\drone_det_env\Scripts\Activate.ps1  # Windows
source drone_det_env/bin/activate      # Linux/macOS

# Update dependencies (if requirements.txt changed)
pip install -r requirements.txt --upgrade
```

---

## 🗑️ Cleaning Up

To remove the virtual environment and start fresh:

**Windows:**
```powershell
Remove-Item drone_det_env -Recurse -Force
python setup.py
```

**Linux/macOS:**
```bash
rm -rf drone_det_env
python3 setup.py
```

---

## 📚 Additional Resources

- **Full Documentation**: See `README.md`
- **Training Guide**: See `TRAINING_FIX_GUIDE.md`
- **Troubleshooting**: See `ALL_ISSUES_AND_FIXES.md`
- **CUDA Issues**: Run `python fix_cuda_issues.py`

---

## 🆘 Getting Help

If you encounter issues:

1. Check the logs: `logs/drone_detection.log`
2. Run diagnostics: `python fix_cuda_issues.py`
3. Verify setup: `python test_training_fix.py`
4. Check this guide and other documentation files

---

## ✅ Checklist for New Users

- [ ] Python 3.8+ installed
- [ ] Repository cloned from git
- [ ] Ran `setup.py` (or `setup.bat`/`setup.sh`)
- [ ] Virtual environment created successfully
- [ ] All dependencies installed
- [ ] Virtual environment activated
- [ ] Dataset configured in `config/config.yaml`
- [ ] Ran `python main.py --mode prepare`
- [ ] Ready to train!

---

**Welcome to the Drone Detection System!** 🎉

If you've completed these steps, you're ready to start training your drone detection model.

Happy training! 🚁

