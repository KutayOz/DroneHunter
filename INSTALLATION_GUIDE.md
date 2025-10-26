# Installation Guide for Drone Detection System

## Prerequisites

### Python Version
- **Required**: Python 3.12
- This project is tested and optimized for Python 3.12
- Other versions are not officially supported and may have compatibility issues

### System Requirements
- Windows 10/11 (64-bit)
- 8GB+ RAM (16GB+ recommended)
- NVIDIA GPU with CUDA support (recommended)
- 10GB+ free disk space

## Installation Steps

### 1. Python Environment Setup

**Option A: Use Python 3.12 (Required)**
```bash
# Download Python 3.12 from https://www.python.org/downloads/
# Install with "Add Python to PATH" checked
# Verify installation
python --version  # Should show Python 3.12.x
```

**Option B: Use Virtual Environment**
```bash
# Create virtual environment
python -m venv drone_detection_env

# Activate virtual environment
# On Windows:
drone_detection_env\Scripts\activate

# On Linux/Mac:
source drone_detection_env/bin/activate
```

### 2. Install Dependencies

**Step 1: Install Core Dependencies**
```bash
pip install --upgrade pip
pip install wheel setuptools
```

**Step 2: Install PyTorch (with CUDA support)**
```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only (slower)
pip install torch torchvision torchaudio
```

**Step 3: Install Computer Vision Libraries**
```bash
pip install opencv-python
pip install Pillow
pip install albumentations
```

**Step 4: Install YOLOv11 and ML Libraries**
```bash
pip install ultralytics
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
```

**Step 5: Install GUI and Utilities**
```bash
pip install customtkinter
pip install loguru
pip install tqdm
pip install psutil
pip install GPUtil
```

**Step 6: Install Additional Dependencies**
```bash
pip install PyYAML
pip install pandas
pip install wandb  # Optional: for experiment tracking
pip install tensorboard  # Optional: for training visualization
```

### 3. Verify Installation

**Test Script:**
```python
# test_installation.py
def test_imports():
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
    
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"✗ OpenCV: {e}")
    
    try:
        import ultralytics
        print(f"✓ Ultralytics {ultralytics.__version__}")
    except ImportError as e:
        print(f"✗ Ultralytics: {e}")
    
    try:
        import customtkinter as ctk
        print("✓ CustomTkinter")
    except ImportError as e:
        print(f"✗ CustomTkinter: {e}")

if __name__ == "__main__":
    test_imports()
```

**Run Test:**
```bash
python test_installation.py
```

### 4. Alternative Installation Methods

**Method 1: Conda Environment**
```bash
# Create conda environment
conda create -n drone_detection python=3.12
conda activate drone_detection

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other packages
pip install -r requirements.txt
```

**Method 2: Docker (Advanced)**
```dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py", "--mode", "gui"]
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size in config/config.yaml
# model:
#   batch_size: 8  # Instead of 16
```

**2. Module Not Found Errors**
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall package
pip uninstall package_name
pip install package_name
```

**3. GUI Not Starting**
```bash
# Install GUI dependencies
pip install customtkinter tkinter-tooltip

# Check display settings
# On Windows: Check display scaling
```

**4. Dataset Loading Issues**
```bash
# Verify dataset structure
python -c "
from pathlib import Path
dataset = Path('Datasets/drone.v1i.yolov11')
print('Dataset exists:', dataset.exists())
print('Train images:', len(list((dataset/'train/images').glob('*'))))
"
```

### Performance Optimization

**1. GPU Memory Optimization**
- Use smaller model size (n, s instead of m, l, x)
- Reduce batch size
- Use mixed precision training
- Close other GPU applications

**2. CPU Optimization**
- Use multiple workers for data loading
- Enable pin_memory for DataLoader
- Use SSD storage for datasets

**3. Training Optimization**
- Use learning rate scheduling
- Implement early stopping
- Use data augmentation
- Monitor GPU utilization

## Quick Start After Installation

1. **Verify Installation:**
   ```bash
   python simple_test.py
   ```

2. **Start GUI:**
   ```bash
   python main.py --mode gui
   ```

3. **Prepare Dataset:**
   ```bash
   python main.py --mode prepare
   ```

4. **Train Model:**
   ```bash
   python main.py --mode train
   ```

5. **Test Detection:**
   ```bash
   python main.py --mode detect --model runs/train/drone_detection/weights/best.pt --input test_image.jpg
   ```

## Support

If you encounter issues:

1. Check Python version (should be 3.12)
2. Verify all dependencies are installed
3. Check GPU drivers and CUDA installation
4. Review log files in `logs/` directory
5. Test with simple_test.py first

For additional help, check the README.md file or create an issue with:
- Python version
- Operating system
- Error messages
- System specifications
