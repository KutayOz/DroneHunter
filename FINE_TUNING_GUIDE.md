# Fine-Tuning Guide for Drone Detection System

## Overview

This guide provides comprehensive instructions for fine-tuning your drone detection model to achieve optimal performance. The system uses YOLOv11 with extensive customization options for different use cases and environments.

## 🎯 Fine-Tuning Strategies

### 1. Hyperparameter Optimization

#### Learning Rate Tuning
```yaml
# config/config.yaml
model:
  lr0: 0.01          # Initial learning rate
  lrf: 0.01          # Final learning rate (lr0 * lrf)
  warmup_epochs: 3.0 # Warmup epochs
  warmup_momentum: 0.8
  warmup_bias_lr: 0.1
```

**Recommended Learning Rate Schedules:**
- **High Learning Rate**: 0.01 → 0.001 (for large datasets)
- **Medium Learning Rate**: 0.005 → 0.0005 (balanced approach)
- **Low Learning Rate**: 0.001 → 0.0001 (for fine-tuning pre-trained models)

**Learning Rate Finder:**
```python
# Add to model_training.py
def find_optimal_lr(model, data_loader, start_lr=1e-7, end_lr=1e-1, num_iter=100):
    """Find optimal learning rate using learning rate finder."""
    lr_finder = torch.optim.lr_scheduler.OneCycleLR(
        model.optimizer, max_lr=end_lr, total_steps=num_iter
    )
    # Implementation details...
```

#### Batch Size Optimization
```yaml
# Auto-adjustment based on GPU memory
# RTX 4090 (24GB): batch_size: 32
# RTX 4080 (16GB): batch_size: 24
# RTX 4070 Ti (12GB): batch_size: 16
# RTX 4060 Ti (8GB): batch_size: 12
# CPU: batch_size: 8
```

**Dynamic Batch Size:**
```python
def get_optimal_batch_size(model_size="m", gpu_memory_gb=16):
    """Calculate optimal batch size based on GPU memory."""
    base_sizes = {"n": 32, "s": 24, "m": 16, "l": 12, "x": 8}
    memory_multiplier = gpu_memory_gb / 16.0
    return int(base_sizes[model_size] * memory_multiplier)
```

### 2. Model Architecture Tuning

#### Model Size Selection
```yaml
# Choose based on speed vs accuracy trade-off
model:
  size: "m"  # Options: n, s, m, l, x
```

**Performance Characteristics:**
- **YOLOv11n**: 100-150 FPS, mAP@0.5: 0.75-0.80
- **YOLOv11s**: 80-120 FPS, mAP@0.5: 0.78-0.83
- **YOLOv11m**: 60-80 FPS, mAP@0.5: 0.80-0.85
- **YOLOv11l**: 30-40 FPS, mAP@0.5: 0.83-0.88
- **YOLOv11x**: 20-30 FPS, mAP@0.5: 0.85-0.90

#### Custom Architecture Modifications
```python
# Custom YOLOv11 configuration
def create_custom_yolo_config():
    """Create custom YOLOv11 configuration for drone detection."""
    config = {
        "backbone": {
            "type": "CSPDarknet",
            "depth_multiple": 0.67,  # Adjust for model size
            "width_multiple": 0.75,  # Adjust for model width
        },
        "neck": {
            "type": "PANet",
            "depth_multiple": 0.67,
            "width_multiple": 0.75,
        },
        "head": {
            "type": "YOLOv11Head",
            "num_classes": 1,  # Only drone class
            "anchors": [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
        }
    }
    return config
```

### 3. Data Augmentation Strategies

#### Basic Augmentation
```yaml
# config/config.yaml
dataset:
  augmentation:
    enabled: true
    hsv_h: 0.015      # Hue variation (0-0.1)
    hsv_s: 0.7        # Saturation variation (0-1.0)
    hsv_v: 0.4        # Value variation (0-1.0)
    degrees: 0.0       # Rotation degrees (0-180)
    translate: 0.1     # Translation (0-1.0)
    scale: 0.5         # Scale variation (0-1.0)
    shear: 0.0         # Shear degrees (0-180)
    perspective: 0.0   # Perspective (0-0.001)
    flipud: 0.0        # Vertical flip probability
    fliplr: 0.5        # Horizontal flip probability
    mosaic: 1.0        # Mosaic augmentation probability
    mixup: 0.0         # Mixup augmentation probability
```

#### Advanced Augmentation for Drone Detection
```python
# Custom augmentation pipeline for drones
def create_drone_augmentation_pipeline():
    """Create specialized augmentation for drone detection."""
    return A.Compose([
        # Geometric transformations
        A.Rotate(limit=15, p=0.5),  # Small rotations for drones
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),  # Rare for drones
        
        # Color and lighting
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.7
        ),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.5
        ),
        
        # Weather effects (realistic for outdoor drone detection)
        A.RandomRain(
            slant_lower=-10,
            slant_upper=10,
            drop_length=20,
            drop_width=1,
            drop_color=(200, 200, 200),
            blur_value=1,
            brightness_coefficient=0.7,
            rain_type="drizzle",
            p=0.1
        ),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.MotionBlur(blur_limit=3, p=0.3),
        ], p=0.3),
        
        # Perspective and distortion
        A.Perspective(scale=(0.05, 0.1), p=0.2),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.1),
        
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

### 4. Loss Function Optimization

#### Custom Loss Weights
```yaml
# config/config.yaml
model:
  box: 7.5    # Box regression loss weight
  cls: 0.5    # Classification loss weight
  dfl: 1.5    # Distribution focal loss weight
```

#### Focal Loss for Imbalanced Data
```python
def create_focal_loss_config():
    """Create focal loss configuration for drone detection."""
    return {
        "focal_loss": {
            "alpha": 0.25,      # Weighting factor for rare class
            "gamma": 2.0,       # Focusing parameter
            "reduction": "mean"
        },
        "class_weights": [1.0, 2.0],  # Higher weight for drone class
    }
```

### 5. Training Strategies

#### Progressive Training
```python
def progressive_training_strategy():
    """Progressive training strategy for better convergence."""
    stages = [
        {
            "epochs": 20,
            "imgsz": 416,
            "lr0": 0.01,
            "batch_size": 32,
            "augmentation": "light"
        },
        {
            "epochs": 40,
            "imgsz": 512,
            "lr0": 0.005,
            "batch_size": 24,
            "augmentation": "medium"
        },
        {
            "epochs": 40,
            "imgsz": 640,
            "lr0": 0.001,
            "batch_size": 16,
            "augmentation": "heavy"
        }
    ]
    return stages
```

#### Transfer Learning
```python
def transfer_learning_setup():
    """Setup transfer learning for drone detection."""
    return {
        "pretrained_weights": "yolov11m.pt",
        "freeze_backbone": True,      # Freeze backbone initially
        "freeze_epochs": 10,          # Freeze for first 10 epochs
        "unfreeze_lr": 0.001,         # Lower LR when unfreezing
        "fine_tune_epochs": 50,       # Fine-tune for 50 epochs
    }
```

### 6. Validation and Testing

#### Cross-Validation Strategy
```python
def k_fold_cross_validation(dataset_path, k=5):
    """Implement k-fold cross-validation for robust evaluation."""
    # Split dataset into k folds
    # Train on k-1 folds, validate on 1 fold
    # Repeat k times
    # Average results across all folds
    pass
```

#### Performance Metrics
```python
def comprehensive_evaluation_metrics():
    """Define comprehensive evaluation metrics."""
    return {
        "primary_metrics": [
            "mAP@0.5",      # Mean Average Precision at IoU 0.5
            "mAP@0.5:0.95", # Mean Average Precision at IoU 0.5:0.95
            "precision",     # Precision
            "recall",        # Recall
            "f1_score"       # F1 Score
        ],
        "secondary_metrics": [
            "inference_time",    # Inference speed
            "model_size",        # Model file size
            "gpu_memory",        # GPU memory usage
            "cpu_usage"          # CPU usage
        ],
        "drone_specific_metrics": [
            "small_drone_detection",    # Detection of small drones
            "occluded_drone_detection", # Detection of partially occluded drones
            "multi_drone_detection",    # Detection of multiple drones
            "false_positive_rate"       # False positive rate
        ]
    }
```

### 7. Environment-Specific Tuning

#### Indoor vs Outdoor Detection
```yaml
# Indoor detection configuration
indoor_config:
  model:
    conf_threshold: 0.3    # Higher confidence for indoor
    iou_threshold: 0.4     # Lower IoU for indoor
  augmentation:
    hsv_v: 0.2            # Less brightness variation
    degrees: 5.0           # Less rotation
    scale: 0.3             # Less scaling

# Outdoor detection configuration
outdoor_config:
  model:
    conf_threshold: 0.25   # Lower confidence for outdoor
    iou_threshold: 0.45    # Standard IoU
  augmentation:
    hsv_v: 0.4            # More brightness variation
    degrees: 15.0          # More rotation
    scale: 0.5             # More scaling
    weather_effects: true  # Add weather effects
```

#### Day vs Night Detection
```python
def day_night_adaptation():
    """Adapt model for day and night conditions."""
    return {
        "day_detection": {
            "conf_threshold": 0.25,
            "augmentation": "standard",
            "color_space": "BGR"
        },
        "night_detection": {
            "conf_threshold": 0.2,     # Lower confidence for night
            "augmentation": "enhanced_contrast",
            "color_space": "HSV",      # Better for low light
            "preprocessing": "histogram_equalization"
        }
    }
```

### 8. Real-time Optimization

#### Inference Speed Optimization
```python
def optimize_inference_speed():
    """Optimize model for real-time inference."""
    return {
        "model_optimization": {
            "half_precision": True,        # Use FP16
            "tensorrt": True,              # Use TensorRT if available
            "onnx_export": True,           # Export to ONNX
            "quantization": "int8"         # Quantize to INT8
        },
        "inference_optimization": {
            "batch_inference": True,       # Process multiple frames
            "async_processing": True,      # Asynchronous processing
            "gpu_pipeline": True,          # GPU pipeline optimization
            "memory_pool": True            # Memory pool for efficiency
        }
    }
```

#### Memory Optimization
```python
def memory_optimization_strategies():
    """Strategies for memory optimization."""
    return {
        "model_compression": {
            "pruning": 0.1,               # Remove 10% of weights
            "knowledge_distillation": True, # Use smaller teacher model
            "quantization": "dynamic"      # Dynamic quantization
        },
        "inference_optimization": {
            "streaming": True,             # Stream processing
            "frame_skipping": 2,           # Process every 2nd frame
            "roi_detection": True,         # Region of interest detection
            "temporal_consistency": True   # Use temporal information
        }
    }
```

### 9. Advanced Fine-Tuning Techniques

#### Multi-Scale Training
```python
def multi_scale_training():
    """Multi-scale training for better generalization."""
    scales = [320, 416, 512, 608, 704, 800]
    return {
        "scale_schedule": {
            "epoch_0_20": scales[:3],     # Small scales initially
            "epoch_20_40": scales[1:4],   # Medium scales
            "epoch_40_60": scales[2:5],   # Large scales
            "epoch_60_80": scales[3:6],   # Very large scales
        },
        "scale_probability": 0.5,         # Probability of scale change
    }
```

#### Adversarial Training
```python
def adversarial_training():
    """Adversarial training for robustness."""
    return {
        "adversarial_epochs": 10,         # Number of adversarial epochs
        "adversarial_lr": 0.001,          # Learning rate for adversarial
        "perturbation_epsilon": 0.03,     # Perturbation magnitude
        "adversarial_weight": 0.1,        # Weight of adversarial loss
    }
```

### 10. Monitoring and Debugging

#### Training Monitoring
```python
def training_monitoring_setup():
    """Setup comprehensive training monitoring."""
    return {
        "metrics_tracking": {
            "wandb": True,                # Weights & Biases logging
            "tensorboard": True,          # TensorBoard logging
            "custom_metrics": True,       # Custom metrics
        },
        "visualization": {
            "loss_curves": True,          # Loss curve plots
            "gradient_flow": True,        # Gradient flow analysis
            "activation_maps": True,      # Activation map visualization
            "detection_samples": True,    # Sample detection results
        },
        "alerts": {
            "loss_spike": True,           # Alert on loss spikes
            "gradient_explosion": True,   # Alert on gradient explosion
            "overfitting": True,          # Alert on overfitting
        }
    }
```

#### Debugging Tools
```python
def debugging_tools():
    """Tools for debugging training issues."""
    return {
        "gradient_checking": {
            "check_gradients": True,      # Check gradient flow
            "gradient_norm": True,        # Monitor gradient norms
            "gradient_clipping": 1.0,     # Gradient clipping value
        },
        "data_debugging": {
            "visualize_augmentations": True,  # Visualize data augmentations
            "check_annotations": True,        # Validate annotations
            "class_distribution": True,       # Check class distribution
        },
        "model_debugging": {
            "activation_analysis": True,      # Analyze activations
            "weight_histograms": True,        # Weight histograms
            "layer_outputs": True,           # Layer output analysis
        }
    }
```

## 🚀 Quick Fine-Tuning Recipes

### Recipe 1: High-Accuracy Drone Detection
```yaml
# For maximum accuracy (slower inference)
model:
  size: "l"              # Large model
  epochs: 150            # More epochs
  batch_size: 12         # Smaller batch size
  lr0: 0.005            # Lower learning rate
  lrf: 0.01             # Lower final learning rate
  patience: 30           # More patience

dataset:
  augmentation:
    enabled: true
    hsv_h: 0.02          # More color variation
    hsv_s: 0.8           # More saturation variation
    hsv_v: 0.5           # More brightness variation
    degrees: 10.0         # More rotation
    translate: 0.15       # More translation
    scale: 0.6            # More scaling
    mosaic: 1.0           # Always use mosaic
    mixup: 0.1            # Add mixup
```

### Recipe 2: Real-Time Drone Detection
```yaml
# For real-time inference (faster, lower accuracy)
model:
  size: "n"              # Nano model
  epochs: 80             # Fewer epochs
  batch_size: 32         # Larger batch size
  lr0: 0.01             # Higher learning rate
  lrf: 0.01             # Standard final learning rate
  patience: 20           # Less patience

inference:
  conf_threshold: 0.3    # Higher confidence threshold
  iou_threshold: 0.5     # Higher IoU threshold
  max_det: 100           # Fewer max detections
```

### Recipe 3: Balanced Performance
```yaml
# Balanced speed and accuracy
model:
  size: "m"              # Medium model
  epochs: 100            # Standard epochs
  batch_size: 16         # Standard batch size
  lr0: 0.01             # Standard learning rate
  lrf: 0.01             # Standard final learning rate
  patience: 50           # Standard patience

dataset:
  augmentation:
    enabled: true
    hsv_h: 0.015         # Moderate color variation
    hsv_s: 0.7           # Moderate saturation variation
    hsv_v: 0.4           # Moderate brightness variation
    degrees: 5.0          # Moderate rotation
    translate: 0.1        # Moderate translation
    scale: 0.5            # Moderate scaling
    mosaic: 1.0           # Use mosaic
    mixup: 0.0            # No mixup
```

## 📊 Performance Benchmarking

### Expected Performance Metrics

| Model Size | mAP@0.5 | mAP@0.5:0.95 | FPS (RTX 4080) | Model Size (MB) |
|------------|---------|--------------|----------------|-----------------|
| YOLOv11n   | 0.75-0.80 | 0.45-0.50   | 100-150        | 6.2             |
| YOLOv11s   | 0.78-0.83 | 0.48-0.53   | 80-120         | 21.5            |
| YOLOv11m   | 0.80-0.85 | 0.50-0.55   | 60-80          | 49.7            |
| YOLOv11l   | 0.83-0.88 | 0.52-0.57   | 30-40          | 83.7            |
| YOLOv11x   | 0.85-0.90 | 0.54-0.59   | 20-30          | 136.7           |

### Optimization Targets

- **Real-time Applications**: >30 FPS, mAP@0.5 >0.75
- **High-Accuracy Applications**: mAP@0.5 >0.85, FPS >10
- **Mobile Applications**: Model size <50MB, FPS >20
- **Edge Devices**: Model size <25MB, FPS >15

## 🔧 Troubleshooting Fine-Tuning Issues

### Common Problems and Solutions

1. **Loss Not Decreasing**
   - Check learning rate (too high/low)
   - Verify data quality and annotations
   - Check gradient flow
   - Reduce batch size

2. **Overfitting**
   - Increase data augmentation
   - Reduce model complexity
   - Add regularization
   - Use early stopping

3. **Underfitting**
   - Increase model complexity
   - Train for more epochs
   - Increase learning rate
   - Check data quality

4. **Slow Convergence**
   - Use learning rate scheduling
   - Implement warmup
   - Check data preprocessing
   - Use transfer learning

5. **Poor Generalization**
   - Increase dataset diversity
   - Use cross-validation
   - Implement data augmentation
   - Check for data leakage

## 📈 Next Steps

1. **Start with Recipe 3** (Balanced Performance)
2. **Monitor training metrics** closely
3. **Adjust hyperparameters** based on validation results
4. **Test on different environments** (indoor/outdoor, day/night)
5. **Optimize for your specific use case**
6. **Deploy and monitor** real-world performance

Remember: Fine-tuning is an iterative process. Start with the recommended settings and adjust based on your specific requirements and performance observations.
