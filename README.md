# Drone Detection System using YOLOv11

A comprehensive drone detection system built with YOLOv11 and Python, featuring GPU-accelerated training, real-time inference, and a modern GUI interface.

## 🚁 Features

- **YOLOv11 Integration**: State-of-the-art object detection for drone identification
- **GPU Acceleration**: CUDA support for fast training and inference
- **Modular Architecture**: Clean, maintainable code structure
- **GUI Interface**: Modern graphical user interface for easy operation
- **Real-time Detection**: Live camera feed and video processing
- **Comprehensive Evaluation**: Detailed performance metrics and analysis
- **Data Augmentation**: Advanced augmentation techniques for better training
- **Multi-dataset Support**: Compatible with multiple drone datasets

## 📋 Requirements

- Python 3.12 (required)
- CUDA-compatible GPU (recommended)
- Windows 10/11 (tested on Windows 10)
- 8GB+ RAM (16GB+ recommended)
- 10GB+ free disk space

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd drone-detection
```

### 2. Automatic Setup (Recommended)

**Windows:**
```powershell
# Option 1: Double-click setup.bat
# Option 2: Run in PowerShell
python setup.py
```

**Linux/macOS:**
```bash
# Make script executable and run
chmod +x setup.sh
./setup.sh

# OR run with Python
python3 setup.py
```

The setup script will:
- ✅ Create a virtual environment (`venv312`)
- ✅ Install PyTorch with CUDA 11.8 support (for GPU acceleration)
- ✅ Install all other dependencies
- ✅ Verify the installation and GPU availability
- ✅ Guide you through next steps

**⏱️ Takes ~10-15 minutes** depending on your internet speed (PyTorch is ~2-3 GB).

### 3. Activate Virtual Environment

**Every time you work on the project**, activate the virtual environment:

**Windows:**
```powershell
.\venv312\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
source venv312/bin/activate
```

You'll see `(venv312)` in your prompt when activated.

> **💡 Tip**: The main.py script will warn you if you forget to activate it!

### 4. Dataset Preparation

> **📦 Note**: Datasets are not included in the repository due to their large size. You need to download and prepare your own drone detection dataset.

**Download Dataset:**
- [Roboflow Drone Detection Dataset](https://universe.roboflow.com/search?q=drone)
- [Kaggle Drone Datasets](https://www.kaggle.com/search?q=drone+detection)
- Or use your own custom dataset

**Dataset Structure:**

```
Datasets/
├── drone.v1i.yolov11/          # Your dataset name
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
```

**Prepare Dataset:**
```bash
python main.py --mode prepare
```

### 3. Run the System

#### GUI Mode (Recommended)
```bash
python main.py --mode gui
```

#### Command Line Mode
```bash
# Prepare dataset
python main.py --mode prepare

# Train model
python main.py --mode train

# Evaluate model
python main.py --mode evaluate --model runs/train/drone_detection/weights/best.pt

# Detect drones in image
python main.py --mode detect --model runs/train/drone_detection/weights/best.pt --input image.jpg --output detected_image.jpg

# Detect drones in video
python main.py --mode detect --model runs/train/drone_detection/weights/best.pt --input video.mp4 --output detected_video.mp4
```

## 🏗️ Project Structure

```
Drone Hashirası/
├── config/
│   └── config.yaml              # Configuration file
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # Dataset preparation and validation
│   ├── model_training.py        # YOLOv11 training with GPU support
│   ├── model_evaluation.py      # Model evaluation and testing
│   ├── inference.py             # Real-time detection and inference
│   └── gui.py                   # Graphical user interface
├── Datasets/                    # Dataset directories
├── logs/                        # Log files
├── runs/                        # Training outputs
├── main.py                      # Main application entry point
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## ⚙️ Configuration

The system is configured via `config/config.yaml`. Key parameters:

### Model Configuration
```yaml
model:
  size: "m"              # Model size: n, s, m, l, x
  epochs: 100            # Training epochs
  batch_size: 16         # Batch size (auto-adjusted for GPU)
  device: "0"            # GPU device (0 for first GPU)
  imgsz: 640             # Input image size
```

### Training Configuration
```yaml
training:
  project: "runs/train"
  name: "drone_detection"
  patience: 50           # Early stopping patience
  save_period: 10        # Save checkpoint every N epochs
```

### Inference Configuration
```yaml
inference:
  conf_threshold: 0.25   # Confidence threshold
  iou_threshold: 0.45    # IoU threshold for NMS
  max_det: 1000          # Maximum detections
```

## 🎯 Usage Guide

### 1. Training a Model

1. **Start the GUI**: `python main.py --mode gui`
2. **Go to Training tab**
3. **Select dataset**: Choose your dataset directory
4. **Configure parameters**: Set epochs, batch size, model size
5. **Validate dataset**: Click "Validate Dataset" to check data integrity
6. **Start training**: Click "Start Training"

The system will:
- Automatically detect GPU availability
- Adjust batch size based on GPU memory
- Monitor training progress with real-time metrics
- Save checkpoints and best model weights
- Generate training plots and logs

### 2. Evaluating a Model

1. **Go to Evaluation tab**
2. **Select model**: Choose the trained model file
3. **Run evaluation**: Click "Evaluate Model" for validation metrics
4. **Test on images**: Click "Test on Images" for sample testing

The system will provide:
- mAP@0.5 and mAP@0.5:0.95 scores
- Precision, Recall, and F1-Score
- Detection statistics and confidence analysis
- Performance plots and detailed reports

### 3. Real-time Detection

1. **Go to Detection tab**
2. **Load model**: Select and load your trained model
3. **Adjust parameters**: Set confidence and IoU thresholds
4. **Choose detection mode**:
   - **Single Image**: Process individual images
   - **Video**: Process video files
   - **Live Camera**: Real-time camera feed

### 4. Batch Processing

For processing multiple images:
```python
from src.inference import DroneDetector, BatchProcessor

# Initialize detector
detector = DroneDetector("path/to/model.pt", config)

# Process multiple images
processor = BatchProcessor(detector)
results = processor.process_images(["image1.jpg", "image2.jpg", "image3.jpg"])
```

## 🔧 Advanced Configuration

### GPU Memory Optimization

The system automatically adjusts batch size based on GPU memory:
- RTX 4090 (24GB): Batch size 32
- RTX 4080 (16GB): Batch size 24
- RTX 4070 Ti (12GB): Batch size 16
- RTX 4060 Ti (8GB): Batch size 12

### Data Augmentation

Configure augmentation in `config/config.yaml`:
```yaml
dataset:
  augmentation:
    enabled: true
    hsv_h: 0.015      # Hue variation
    hsv_s: 0.7        # Saturation variation
    hsv_v: 0.4        # Value variation
    degrees: 0.0       # Rotation
    translate: 0.1     # Translation
    scale: 0.5         # Scaling
    fliplr: 0.5        # Horizontal flip
```

### Custom Model Architecture

To use different YOLOv11 sizes:
- **n**: Nano (fastest, least accurate)
- **s**: Small (fast, good accuracy)
- **m**: Medium (balanced, recommended)
- **l**: Large (slower, higher accuracy)
- **x**: Extra Large (slowest, highest accuracy)

## 📊 Performance Monitoring

The system provides comprehensive monitoring:

### Training Metrics
- Loss curves (training and validation)
- Precision and Recall over time
- mAP@0.5 and mAP@0.5:0.95 progression
- GPU utilization and memory usage
- Training speed (FPS)

### Detection Statistics
- Total detections per frame
- Confidence score distribution
- Detection count per image
- Processing speed and FPS

### System Resources
- GPU memory usage
- GPU temperature
- CPU utilization
- RAM usage

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in config
   - Use smaller model size (n or s)
   - Close other GPU applications

2. **Dataset Not Found**
   - Check dataset path in config
   - Ensure proper directory structure
   - Run dataset validation

3. **Model Loading Failed**
   - Verify model file exists
   - Check model compatibility
   - Ensure proper file permissions

4. **GUI Not Starting**
   - Install customtkinter: `pip install customtkinter`
   - Check Python version (3.12)
   - Update graphics drivers

### Log Files

Check logs in `logs/drone_detection.log` for detailed error information.

## 🔄 Fine-tuning Guide

### 1. Hyperparameter Tuning

**Learning Rate**:
- Start with 0.01
- Reduce by factor of 10 if loss plateaus
- Use learning rate scheduling

**Batch Size**:
- Increase for better gradient estimates
- Decrease if GPU memory insufficient
- Use powers of 2 (8, 16, 32, 64)

**Epochs**:
- Start with 100 epochs
- Use early stopping (patience=50)
- Monitor validation loss for overfitting

### 2. Data Augmentation

**For Small Datasets**:
- Increase augmentation intensity
- Use more aggressive transformations
- Add noise and blur effects

**For Large Datasets**:
- Reduce augmentation to prevent overfitting
- Focus on geometric transformations
- Use color space augmentations sparingly

### 3. Model Architecture

**Speed vs Accuracy Trade-off**:
- Use YOLOv11n for real-time applications
- Use YOLOv11m for balanced performance
- Use YOLOv11l/x for maximum accuracy

### 4. Training Strategies

**Transfer Learning**:
- Start with pretrained weights
- Freeze backbone layers initially
- Gradually unfreeze and fine-tune

**Progressive Training**:
- Start with smaller images (416x416)
- Gradually increase to full size (640x640)
- Use different learning rates for different stages

### 5. Evaluation and Validation

**Cross-Validation**:
- Use multiple dataset splits
- Validate on different environments
- Test on edge cases and difficult scenarios

**Performance Metrics**:
- Focus on mAP@0.5 for drone detection
- Monitor precision-recall curves
- Analyze false positive/negative patterns

### 6. Post-Processing Optimization

**Confidence Thresholding**:
- Adjust based on application requirements
- Higher threshold = fewer false positives
- Lower threshold = more detections

**Non-Maximum Suppression**:
- Tune IoU threshold (0.45 default)
- Adjust based on drone density
- Consider temporal consistency for videos

## 📈 Expected Performance

### Training Time (RTX 4080)
- YOLOv11n: ~2-3 hours (100 epochs)
- YOLOv11m: ~4-6 hours (100 epochs)
- YOLOv11l: ~8-12 hours (100 epochs)

### Inference Speed
- YOLOv11n: ~100-150 FPS
- YOLOv11m: ~60-80 FPS
- YOLOv11l: ~30-40 FPS

### Accuracy (Expected mAP@0.5)
- YOLOv11n: 0.75-0.80
- YOLOv11m: 0.80-0.85
- YOLOv11l: 0.85-0.90

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv11
- [Roboflow](https://roboflow.com/) for drone datasets
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) for modern GUI
- [Albumentations](https://github.com/albumentations-team/albumentations) for data augmentation

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review log files
3. Create an issue with detailed information
4. Include system specifications and error messages

---

**Happy Drone Detecting! 🚁✨**
