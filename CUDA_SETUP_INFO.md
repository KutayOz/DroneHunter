# CUDA Setup Information

## 🎯 Automatic CUDA Installation

This project automatically installs PyTorch with CUDA support during setup to ensure GPU acceleration works out of the box.

## 📦 What Gets Installed

When you run `setup.py`, it automatically installs:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

This ensures:
- ✅ PyTorch with CUDA 11.8 support
- ✅ GPU acceleration enabled by default
- ✅ No manual CUDA configuration needed
- ✅ Works with most NVIDIA GPUs (GTX 10 series and newer)

## 🚀 Why CUDA 11.8?

- **Best compatibility** with most systems
- **Stable** and well-tested
- **Works with** RTX 20, 30, and 40 series GPUs
- **Smaller download** than CUDA 12.x
- **Recommended** by PyTorch for production use

## 🔍 Verification

After setup, the script automatically verifies:

```
PyTorch Information:
  Version: 2.x.x+cu118
  CUDA Available: True
  CUDA Version: 11.8
  GPU Count: 1
  GPU 0: NVIDIA GeForce RTX 3060 Laptop GPU
```

If you see `CUDA Available: True`, you're ready to train with GPU acceleration! 🎉

## 🐛 Troubleshooting

### CUDA Not Available After Setup

If `CUDA Available: False` after setup:

1. **Check GPU Drivers**
   ```powershell
   nvidia-smi
   ```
   If this fails, install/update NVIDIA drivers from:
   https://www.nvidia.com/Download/index.aspx

2. **Restart Computer**
   Sometimes Windows needs a restart after driver installation.

3. **Verify Installation**
   ```powershell
   .\venv312\Scripts\Activate.ps1
   python -c "import torch; print('CUDA:', torch.cuda.is_available())"
   ```

4. **Manual Reinstall** (if needed)
   ```powershell
   .\venv312\Scripts\Activate.ps1
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Different CUDA Version Needed

If you need a different CUDA version:

**CUDA 12.1** (newer, requires updated drivers):
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CPU Only** (not recommended, very slow):
```powershell
pip install torch torchvision torchaudio
```

## 📊 Performance Impact

### With GPU (CUDA):
- Training: **60-80 FPS**
- Epoch time: **4-6 minutes**
- 100 epochs: **~8-10 hours**

### Without GPU (CPU only):
- Training: **0.1-0.5 FPS**
- Epoch time: **60-120 minutes**
- 100 epochs: **100-200 hours**

**GPU is 100-200x faster!** 🚀

## 🎓 Technical Details

### What is CUDA?

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform that allows PyTorch to use your GPU for training neural networks.

### Why Separate Installation?

PyTorch with CUDA is installed separately from `requirements.txt` because:
1. **Size**: PyTorch with CUDA is ~2-3 GB
2. **Flexibility**: Users can choose CPU-only if needed
3. **Compatibility**: Different systems may need different CUDA versions
4. **Speed**: Installing from CUDA-specific index is faster

### Index URL Explained

```bash
--index-url https://download.pytorch.org/whl/cu118
```

This tells pip to download PyTorch from the CUDA 11.8 specific repository, ensuring you get the GPU-enabled version instead of the CPU-only default.

## ✅ Supported GPUs

This setup works with:
- ✅ RTX 40 series (4090, 4080, 4070, 4060)
- ✅ RTX 30 series (3090, 3080, 3070, 3060)
- ✅ RTX 20 series (2080 Ti, 2080, 2070, 2060)
- ✅ GTX 16 series (1660 Ti, 1660, 1650)
- ✅ GTX 10 series (1080 Ti, 1080, 1070, 1060)
- ✅ Quadro and Tesla GPUs

### Minimum Requirements
- **Compute Capability**: 3.5 or higher
- **VRAM**: 4GB minimum (6GB+ recommended)
- **Driver**: 450.80.02 or newer

## 🔄 Updating PyTorch

To update to the latest PyTorch version:

```powershell
.\venv312\Scripts\Activate.ps1
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📝 Manual Installation

If you prefer to install manually:

1. **Skip automatic setup** or install without CUDA first
2. **Activate virtual environment**
3. **Install PyTorch with CUDA**:
   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## 🌐 Alternative CUDA Versions

### CUDA 12.1 (Latest)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
- Newer features
- Better performance on RTX 40 series
- Requires driver 530+ on Linux, 528+ on Windows

### CUDA 11.8 (Recommended)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
- Best compatibility
- Stable and tested
- Works with older drivers

### CPU Only
```bash
pip install torch torchvision torchaudio
```
- No GPU required
- Very slow for training
- Use only for testing or inference

## 🎯 Best Practices

1. **Always use GPU** for training (100x faster)
2. **Keep drivers updated** for best performance
3. **Monitor GPU temperature** during training
4. **Close other GPU applications** before training
5. **Use appropriate batch size** for your GPU memory

## 📞 Getting Help

If you encounter CUDA issues:

1. **Check logs**: `logs/drone_detection.log`
2. **Run diagnostics**: `python main.py --mode info`
3. **Verify GPU**: `nvidia-smi`
4. **Test PyTorch**: `python -c "import torch; print(torch.cuda.is_available())"`
5. **Create an issue** on GitHub with error details

## 🔗 Useful Links

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
- [CUDA Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)
- [PyTorch CUDA Wheels](https://download.pytorch.org/whl/torch_stable.html)

---

**Remember**: GPU acceleration is automatically configured by `setup.py` - you don't need to do anything special! 🎉

