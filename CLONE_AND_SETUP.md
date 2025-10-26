# Clone and Setup Instructions

## For Team Members / New Users

If you're cloning this project from Git for the first time, follow these instructions:

---

## 📥 Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd drone-detection
```

Replace `<your-repository-url>` with your actual Git repository URL.

---

## ⚙️ Step 2: Run Automatic Setup

The project includes an automatic setup system that will:
- Create a virtual environment specific to your machine
- Install all dependencies
- Verify everything works

### On Windows

**Method 1: Double-click**
- Find and double-click `setup.bat` in the project folder

**Method 2: PowerShell**
```powershell
python setup.py
```

### On Linux/macOS

**Method 1: Shell script**
```bash
chmod +x setup.sh
./setup.sh
```

**Method 2: Python**
```bash
python3 setup.py
```

### What Happens During Setup?

```
1. Checking Python version... ✓
2. Creating virtual environment (drone_det_env)... ✓
3. Upgrading pip... ✓
4. Installing torch... ⏳
5. Installing ultralytics... ⏳
6. Installing opencv... ⏳
7. Installing other dependencies... ⏳
8. Verifying installation... ✓
9. Done! ✓
```

⏱️ **Time**: 5-10 minutes (depending on internet speed)

---

## 🎯 Step 3: Activate Virtual Environment

**IMPORTANT**: You must activate the virtual environment **every time** you work on the project.

### Windows
```powershell
.\drone_det_env\Scripts\Activate.ps1
```

### Linux/macOS
```bash
source drone_det_env/bin/activate
```

### How to Know It's Activated?

You'll see `(drone_det_env)` at the start of your terminal prompt:
```
(drone_det_env) C:\Users\YourName\drone-detection>
```

---

## 📊 Step 4: Configure Your Dataset

1. Place your dataset in the `Datasets/` folder

2. Edit `config/config.yaml`:
```yaml
dataset:
  primary_dataset: "Datasets/your-dataset-name"
```

---

## ✅ Step 5: Verify Setup

Run the verification script:
```bash
python test_training_fix.py
```

You should see all checks passing:
```
✓ Module Imports     : PASS
✓ data.yaml         : PASS
✓ config.yaml       : PASS
✓ CUDA Setup        : PASS
✓ Model File        : PASS
```

---

## 🚀 Step 6: Start Using the System

### Prepare Dataset
```bash
python main.py --mode prepare
```

### Train Model
```bash
python main.py --mode train
```

### Use GUI
```bash
python main.py --mode gui
```

---

## 🔄 Daily Workflow

Every time you come back to work on the project:

1. **Open terminal in project directory**

2. **Activate virtual environment**
   ```bash
   # Windows
   .\drone_det_env\Scripts\Activate.ps1
   
   # Linux/macOS
   source drone_det_env/bin/activate
   ```

3. **Work on the project**
   ```bash
   python main.py --mode train
   ```

4. **When done, deactivate** (optional)
   ```bash
   deactivate
   ```

---

## 🐛 Troubleshooting

### "python: command not found"
**Fix**: Install Python 3.12 from [python.org](https://www.python.org/downloads/)

### "Cannot activate virtual environment" (Windows)
**Fix**: Enable script execution:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### "Setup failed to install packages"
**Fix**: Try manual installation:
```bash
.\drone_det_env\Scripts\Activate.ps1
pip install -r requirements.txt
```

### "ImportError" when running
**Fix**: Make sure virtual environment is activated! Look for `(drone_det_env)` in your prompt.

### "CUDA out of memory"
**Fix**: Reduce batch size in `config/config.yaml`:
```yaml
model:
  batch_size: 4  # or 2
```

---

## 📁 What's NOT in Git?

These files/folders are NOT tracked by Git (see `.gitignore`):

- ❌ `drone_det_env/` - Virtual environment (each user creates their own)
- ❌ `runs/` - Training outputs
- ❌ `logs/*.log` - Log files
- ❌ `*.pt` - Model weights (large files)
- ❌ `__pycache__/` - Python cache

This is why **each user must run setup.py** after cloning!

---

## 🎓 Understanding the Setup

### Why Virtual Environment?

1. **Isolation**: Prevents conflicts with other Python projects
2. **Reproducibility**: Everyone has the same package versions
3. **Portability**: Each user's environment matches their machine
4. **Clean**: Easy to delete and recreate if something breaks

### Why Not Include venv in Git?

1. ❌ **Large size**: 500MB+ of installed packages
2. ❌ **Platform-specific**: Windows paths don't work on Linux
3. ❌ **User-specific**: Contains absolute paths to your machine
4. ✅ **Better**: Each user creates their own fresh environment

### The Setup Process Creates:

```
drone-detection/
├── drone_det_env/          # YOUR virtual environment
│   ├── Scripts/            # (Windows) or bin/ (Linux)
│   ├── Lib/               # Installed packages
│   └── pyvenv.cfg         # Configuration with YOUR paths
├── ... (rest of project files)
```

---

## 🤝 For Team Collaboration

### When You Make Changes

```bash
# Make changes to code
git add .
git commit -m "Your changes"
git push origin main
```

### When Teammate Pulls Changes

```bash
# Pull latest code
git pull origin main

# Activate venv
.\drone_det_env\Scripts\Activate.ps1

# If requirements.txt changed, update packages
pip install -r requirements.txt --upgrade
```

### Adding New Dependencies

If you add a new package:

1. Install it in your venv:
   ```bash
   pip install new-package
   ```

2. Update requirements.txt:
   ```bash
   pip freeze > requirements.txt
   ```

3. Commit and push:
   ```bash
   git add requirements.txt
   git commit -m "Added new-package dependency"
   git push
   ```

4. Tell teammates to run:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📚 Additional Documentation

- **SETUP_GUIDE.md** - Detailed setup instructions
- **README.md** - Project overview and usage
- **TRAINING_FIX_GUIDE.md** - Training troubleshooting
- **ALL_ISSUES_AND_FIXES.md** - Common issues and solutions

---

## ✅ Setup Checklist

- [ ] Cloned repository
- [ ] Ran `setup.py` (or `setup.bat`/`setup.sh`)
- [ ] Virtual environment created successfully
- [ ] Can activate virtual environment
- [ ] See `(drone_det_env)` in terminal prompt
- [ ] Ran `python test_training_fix.py` - all pass
- [ ] Configured dataset in `config/config.yaml`
- [ ] Ready to train!

---

**🎉 You're all set!** Welcome to the team!

If you encounter any issues not covered here, check the other documentation files or ask for help.

