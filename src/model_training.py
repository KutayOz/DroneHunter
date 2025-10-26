"""
Model Training Module for Drone Detection System

This module handles YOLOv11 model training with GPU support,
monitoring, and checkpoint management.
"""

import os
import yaml
import torch
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import GPUtil
import psutil
from datetime import datetime


class GPUMonitor:
    """
    Monitor GPU usage and memory during training.
    """
    
    def __init__(self):
        """Initialize GPU monitor."""
        self.gpus = GPUtil.getGPUs() if GPUtil.getGPUs() else []
        self.device_count = torch.cuda.device_count()
        
    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get current GPU information.
        
        Returns:
            Dict[str, Any]: GPU information including memory usage and temperature
        """
        if not self.gpus:
            return {"available": False, "message": "No GPUs detected"}
        
        gpu_info = {
            "available": True,
            "count": len(self.gpus),
            "devices": []
        }
        
        for i, gpu in enumerate(self.gpus):
            device_info = {
                "id": gpu.id,
                "name": gpu.name,
                "memory_total": f"{gpu.memoryTotal} MB",
                "memory_used": f"{gpu.memoryUsed} MB",
                "memory_free": f"{gpu.memoryFree} MB",
                "memory_utilization": f"{gpu.memoryUtil * 100:.1f}%",
                "gpu_utilization": f"{gpu.load * 100:.1f}%",
                "temperature": f"{gpu.temperature}°C"
            }
            gpu_info["devices"].append(device_info)
        
        return gpu_info
    
    def check_gpu_availability(self) -> bool:
        """
        Check if GPU is available for training.
        
        Returns:
            bool: True if GPU is available, False otherwise
        """
        return torch.cuda.is_available() and len(self.gpus) > 0
    
    def get_recommended_batch_size(self, model_size: str = "m") -> int:
        """
        Get recommended batch size based on GPU memory.
        
        Args:
            model_size (str): Model size (n, s, m, l, x)
            
        Returns:
            int: Recommended batch size
        """
        if not self.gpus:
            return 8  # Default for CPU
        
        gpu_memory = self.gpus[0].memoryTotal
        
        # Batch size recommendations based on GPU memory and model size
        size_multipliers = {"n": 1, "s": 1.2, "m": 1.5, "l": 2, "x": 2.5}
        multiplier = size_multipliers.get(model_size, 1.5)
        
        if gpu_memory >= 24000:  # RTX 4090, A100
            return int(32 * multiplier)
        elif gpu_memory >= 16000:  # RTX 4080, RTX 3090
            return int(24 * multiplier)
        elif gpu_memory >= 12000:  # RTX 4070 Ti, RTX 3080
            return int(16 * multiplier)
        elif gpu_memory >= 8000:   # RTX 4060 Ti, RTX 3070
            return int(12 * multiplier)
        else:
            return int(8 * multiplier)


class TrainingMonitor:
    """
    Monitor training progress and metrics.
    """
    
    def __init__(self, log_dir: str):
        """
        Initialize training monitor.
        
        Args:
            log_dir (str): Directory to save training logs and plots
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training metrics storage
        self.metrics = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "precision": [],
            "recall": [],
            "mAP50": [],
            "mAP50-95": []
        }
        
        # System metrics
        self.system_metrics = {
            "epoch": [],
            "gpu_memory": [],
            "gpu_utilization": [],
            "cpu_utilization": [],
            "ram_usage": []
        }
    
    def update_metrics(self, epoch: int, results: Dict[str, float]):
        """
        Update training metrics.
        
        Args:
            epoch (int): Current epoch
            results (Dict[str, float]): Training results from YOLO
        """
        self.metrics["epoch"].append(epoch)
        
        # Extract metrics from YOLO results
        if hasattr(results, 'results_dict'):
            results_dict = results.results_dict
            self.metrics["train_loss"].append(results_dict.get("train/box_loss", 0.0))
            self.metrics["val_loss"].append(results_dict.get("val/box_loss", 0.0))
            self.metrics["precision"].append(results_dict.get("metrics/precision(B)", 0.0))
            self.metrics["recall"].append(results_dict.get("metrics/recall(B)", 0.0))
            self.metrics["mAP50"].append(results_dict.get("metrics/mAP50(B)", 0.0))
            self.metrics["mAP50-95"].append(results_dict.get("metrics/mAP50-95(B)", 0.0))
    
    def update_system_metrics(self, epoch: int, gpu_monitor: GPUMonitor):
        """
        Update system metrics.
        
        Args:
            epoch (int): Current epoch
            gpu_monitor (GPUMonitor): GPU monitor instance
        """
        self.system_metrics["epoch"].append(epoch)
        
        # GPU metrics
        gpu_info = gpu_monitor.get_gpu_info()
        if gpu_info["available"] and gpu_info["devices"]:
            gpu = gpu_info["devices"][0]
            self.system_metrics["gpu_memory"].append(float(gpu["memory_used"].split()[0]))
            self.system_metrics["gpu_utilization"].append(float(gpu["gpu_utilization"].rstrip('%')))
        else:
            self.system_metrics["gpu_memory"].append(0)
            self.system_metrics["gpu_utilization"].append(0)
        
        # CPU and RAM metrics
        self.system_metrics["cpu_utilization"].append(psutil.cpu_percent())
        self.system_metrics["ram_usage"].append(psutil.virtual_memory().percent)
    
    def plot_training_metrics(self):
        """Plot training metrics."""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Training Metrics', fontsize=16)
            
            # Loss plot
            axes[0, 0].plot(self.metrics["epoch"], self.metrics["train_loss"], label='Train Loss', color='blue')
            axes[0, 0].plot(self.metrics["epoch"], self.metrics["val_loss"], label='Val Loss', color='red')
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Precision plot
            axes[0, 1].plot(self.metrics["epoch"], self.metrics["precision"], color='green')
            axes[0, 1].set_title('Precision')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].grid(True)
            
            # Recall plot
            axes[0, 2].plot(self.metrics["epoch"], self.metrics["recall"], color='orange')
            axes[0, 2].set_title('Recall')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Recall')
            axes[0, 2].grid(True)
            
            # mAP50 plot
            axes[1, 0].plot(self.metrics["epoch"], self.metrics["mAP50"], color='purple')
            axes[1, 0].set_title('mAP@0.5')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('mAP@0.5')
            axes[1, 0].grid(True)
            
            # mAP50-95 plot
            axes[1, 1].plot(self.metrics["epoch"], self.metrics["mAP50-95"], color='brown')
            axes[1, 1].set_title('mAP@0.5:0.95')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('mAP@0.5:0.95')
            axes[1, 1].grid(True)
            
            # System metrics plot
            axes[1, 2].plot(self.system_metrics["epoch"], self.system_metrics["gpu_utilization"], 
                           label='GPU Util %', color='red')
            axes[1, 2].plot(self.system_metrics["epoch"], self.system_metrics["cpu_utilization"], 
                           label='CPU Util %', color='blue')
            axes[1, 2].set_title('System Utilization')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Utilization %')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
            
            plt.tight_layout()
            plt.savefig(self.log_dir / 'training_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Training metrics plot saved to {self.log_dir / 'training_metrics.png'}")
            
        except Exception as e:
            logger.error(f"Failed to plot training metrics: {e}")
    
    def save_metrics(self):
        """Save metrics to CSV file."""
        try:
            import pandas as pd
            
            # Save training metrics
            training_df = pd.DataFrame(self.metrics)
            training_df.to_csv(self.log_dir / 'training_metrics.csv', index=False)
            
            # Save system metrics
            system_df = pd.DataFrame(self.system_metrics)
            system_df.to_csv(self.log_dir / 'system_metrics.csv', index=False)
            
            logger.info(f"Metrics saved to {self.log_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")


class YOLOv11Trainer:
    """
    YOLOv11 model trainer with GPU support and monitoring.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize YOLOv11 trainer.
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config
        self.model_config = config.get("model", {})
        self.training_config = config.get("training", {})
        
        # Initialize components
        self.gpu_monitor = GPUMonitor()
        self.monitor = None
        
        # Training state
        self.model = None
        self.training_results = None
    
    def validate_gpu_before_training(self) -> bool:
        """
        Validate GPU availability before training and prompt user if GPU is not available.
        
        Returns:
            bool: True if GPU is available or user chooses to continue, False otherwise
        """
        print("\n" + "=" * 80)
        print("GPU VALIDATION CHECK")
        print("=" * 80)
        
        # Check if CUDA is available in PyTorch
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if cuda_available else 0
        
        print(f"\n✓ PyTorch CUDA Available: {cuda_available}")
        print(f"✓ Number of GPUs detected by PyTorch: {gpu_count}")
        
        # Check GPUtil for GPU info
        try:
            gpus = GPUtil.getGPUs()
            print(f"✓ Number of GPUs detected by GPUtil: {len(gpus)}")
            
            if gpus:
                print("\nGPU Information:")
                print("-" * 80)
                for i, gpu in enumerate(gpus):
                    print(f"  GPU {i}: {gpu.name}")
                    print(f"    - Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB used")
                    print(f"    - Utilization: {gpu.load * 100:.1f}%")
                    print(f"    - Temperature: {gpu.temperature}°C")
        except Exception as e:
            print(f"⚠ GPUtil error: {e}")
            gpus = []
        
        # Check if user wants to use GPU
        desired_device = self.model_config.get("device", "0")
        use_gpu = desired_device != "cpu"
        
        print(f"\n→ Desired device (from config): {desired_device}")
        
        # Validation logic
        if use_gpu and not cuda_available:
            print("\n" + "=" * 80)
            print("❌ ERROR: GPU REQUESTED BUT NOT AVAILABLE!")
            print("=" * 80)
            print("\nYour config specifies GPU training, but:")
            print(f"  • PyTorch CUDA is not available")
            print(f"  • Number of GPUs: {gpu_count}")
            print("\nPossible reasons:")
            print("  1. PyTorch was installed without CUDA support")
            print("  2. NVIDIA GPU drivers are not installed")
            print("  3. CUDA version mismatch")
            print("  4. GPU is not detected by the system")
            
            print("\nRecommended actions:")
            print("  1. Reinstall PyTorch with CUDA support:")
            print("     pip uninstall torch torchvision")
            print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            print("\n  2. Or change config/config.yaml to use CPU:")
            print('     Change "device: \"0\"" to "device: \"cpu\""')
            print("\n  3. Run diagnostic script:")
            print("     python fix_cuda_issues.py")
            
            print("\n" + "=" * 80)
            response = input("\n⚠️  Do you want to continue with CPU training? (VERY SLOW!) [y/N]: ").strip().lower()
            
            if response != 'y':
                print("\n❌ Training cancelled by user.")
                print("Please fix GPU issues before training.")
                print("=" * 80 + "\n")
                return False
            else:
                print("\n⚠️  Proceeding with CPU training (will be VERY slow)...")
                print("⚠️  Estimated time: 5-10+ hours per epoch")
                self.model_config["device"] = "cpu"
        
        elif use_gpu and cuda_available and gpu_count == 0:
            print("\n" + "=" * 80)
            print("⚠️  WARNING: PyTorch CUDA is available but no GPUs found!")
            print("=" * 80)
            print("\nThis might indicate:")
            print("  1. GPU is not properly recognized by the system")
            print("  2. Driver issues")
            print("\nRunning 'nvidia-smi' might help diagnose the issue.")
            
            response = input("\n⚠️  Continue anyway? [y/N]: ").strip().lower()
            if response != 'y':
                print("\n❌ Training cancelled.")
                return False
        
        elif use_gpu and cuda_available:
            print("\n" + "=" * 80)
            print("✅ GPU IS AVAILABLE AND READY FOR TRAINING")
            print("=" * 80)
            
            # Display which GPU will be used
            if desired_device.isdigit():
                device_id = int(desired_device)
                if device_id < gpu_count:
                    gpu_name = torch.cuda.get_device_name(device_id)
                    print(f"\n→ Will use GPU {device_id}: {gpu_name}")
                    
                    # Get memory info
                    try:
                        mem_info = torch.cuda.mem_get_info(device_id)
                        free_mem = mem_info[0] / (1024**3)
                        total_mem = mem_info[1] / (1024**3)
                        print(f"→ Available GPU memory: {free_mem:.2f} GB / {total_mem:.2f} GB")
                    except:
                        pass
                else:
                    print(f"\n⚠️  WARNING: Device {device_id} not available (only {gpu_count} GPU(s))")
                    print(f"→ Falling back to GPU 0")
                    self.model_config["device"] = "0"
            else:
                print(f"\n→ Will use all available GPUs")
        
        print("\n" + "=" * 80)
        input("\nPress Enter to start training...")
        print("\n")
        
        return True
        
    def setup_training(self, data_yaml_path: str) -> bool:
        """
        Setup training environment and model.
        
        Args:
            data_yaml_path (str): Path to data.yaml file
            
        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            # Validate GPU before proceeding
            if not self.validate_gpu_before_training():
                logger.error("GPU validation failed. Training cancelled.")
                return False
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    logger.info("CUDA cache cleared")
                except Exception as e:
                    logger.warning(f"Failed to clear CUDA cache: {e}")
            
            # Check GPU availability and adjust settings
            if self.model_config.get("device") != "cpu":
                gpu_info = self.gpu_monitor.get_gpu_info()
                if gpu_info.get("available"):
                    logger.info(f"GPU available: {gpu_info}")
                    
                    # Adjust batch size based on GPU memory
                    recommended_batch_size = self.gpu_monitor.get_recommended_batch_size(
                        self.model_config.get("size", "m")
                    )
                    # Be more conservative with batch size to avoid CUDA errors
                    recommended_batch_size = max(4, recommended_batch_size // 2)
                    logger.info(f"Recommended batch size for GPU: {recommended_batch_size}")
                    
                    if self.model_config.get("batch_size", 16) > recommended_batch_size:
                        logger.warning(f"Reducing batch size from {self.model_config['batch_size']} to {recommended_batch_size}")
                        self.model_config["batch_size"] = recommended_batch_size
                else:
                    logger.warning("GPU not available, using CPU")
                    self.model_config["device"] = "cpu"
            
            # Initialize model
            model_size = self.model_config.get("size", "m")
            model_file = f"yolo11{model_size}.pt"
            
            # Check if model file exists, if not, try to download it
            if not os.path.exists(model_file):
                logger.info(f"Model file {model_file} not found. Attempting to download...")
                try:
                    # Try to download the model
                    from ultralytics import YOLO
                    self.model = YOLO(model_file)  # This will download if not present
                    logger.info(f"Model {model_file} downloaded successfully")
                except Exception as e:
                    logger.error(f"Failed to download model {model_file}: {e}")
                    logger.error("Please install ultralytics: pip install ultralytics")
                    logger.error("Or download the model manually from: https://github.com/ultralytics/assets/releases")
                    return False
            else:
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(model_file)
                except ImportError:
                    logger.error("ultralytics not installed. Please run: pip install ultralytics")
                    return False
                except Exception as e:
                    logger.error(f"Failed to load model {model_file}: {e}")
                    logger.error("The model file may be corrupted. Please download it again.")
                    return False
            
            # Setup monitoring
            project_dir = Path(self.training_config.get("project", "runs/train"))
            run_name = self.training_config.get("name", "drone_detection")
            self.monitor = TrainingMonitor(project_dir / run_name)
            
            logger.info(f"Training setup completed for model size: {model_size}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup training: {e}")
            return False
    
    def train(self, data_yaml_path: str) -> bool:
        """
        Train the YOLOv11 model.
        
        Args:
            data_yaml_path (str): Path to data.yaml file
            
        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            if not self.setup_training(data_yaml_path):
                return False
            
            # Prepare training arguments
            train_args = {
                "data": data_yaml_path,
                "epochs": self.model_config.get("epochs", 100),
                "batch": self.model_config.get("batch_size", 16),
                "imgsz": self.model_config.get("imgsz", 640),
                "device": self.model_config.get("device", "0"),
                "project": self.training_config.get("project", "runs/train"),
                "name": self.training_config.get("name", "drone_detection"),
                "save_period": self.model_config.get("save_period", 10),
                "patience": self.training_config.get("patience", 50),
                "resume": self.training_config.get("resume", False),
                "pretrained": self.model_config.get("pretrained", True),
                "plots": self.model_config.get("plots", True),
                "val": self.model_config.get("val", True),
                
                # Optimization parameters
                "optimizer": self.model_config.get("optimizer", "AdamW"),
                "lr0": self.model_config.get("lr0", 0.01),
                "lrf": self.model_config.get("lrf", 0.01),
                "momentum": self.model_config.get("momentum", 0.937),
                "weight_decay": self.model_config.get("weight_decay", 0.0005),
                "warmup_epochs": self.model_config.get("warmup_epochs", 3.0),
                "warmup_momentum": self.model_config.get("warmup_momentum", 0.8),
                "warmup_bias_lr": self.model_config.get("warmup_bias_lr", 0.1),
                
                # Loss function parameters
                "box": self.model_config.get("box", 7.5),
                "cls": self.model_config.get("cls", 0.5),
                "dfl": self.model_config.get("dfl", 1.5),
            }
            
            logger.info("Starting YOLOv11 training...")
            logger.info(f"Training arguments: {train_args}")
            
            # Start training with CUDA error handling
            start_time = time.time()
            try:
                self.training_results = self.model.train(**train_args)
            except RuntimeError as e:
                error_msg = str(e)
                if "CUDA" in error_msg or "out of memory" in error_msg:
                    logger.error(f"CUDA error encountered: {error_msg}")
                    logger.warning("Attempting to retry with CPU or smaller batch size...")
                    
                    # Clear CUDA cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # If using GPU, try with CPU instead
                    if train_args["device"] != "cpu":
                        logger.info("Retrying training with CPU...")
                        train_args["device"] = "cpu"
                        train_args["batch"] = max(4, train_args["batch"] // 2)
                        logger.info(f"Reduced batch size to {train_args['batch']} for CPU training")
                        
                        # Reinitialize model for CPU
                        model_size = self.model_config.get("size", "m")
                        model_file = f"yolo11{model_size}.pt"
                        self.model = YOLO(model_file)
                        
                        # Retry training on CPU
                        self.training_results = self.model.train(**train_args)
                    else:
                        raise e
                else:
                    raise e
            
            training_time = time.time() - start_time
            
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Save final metrics and plots
            if self.monitor:
                self.monitor.plot_training_metrics()
                self.monitor.save_metrics()
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def get_training_results(self) -> Optional[Dict]:
        """
        Get training results and metrics.
        
        Returns:
            Optional[Dict]: Training results if available
        """
        if self.training_results is None:
            return None
        
        return {
            "model_path": str(self.training_results.save_dir / "weights" / "best.pt"),
            "last_epoch": self.training_results.epoch,
            "best_fitness": self.training_results.best_fitness,
            "results_dict": self.training_results.results_dict if hasattr(self.training_results, 'results_dict') else {}
        }
    
    def validate_model(self, data_yaml_path: str) -> Dict:
        """
        Validate the trained model.
        
        Args:
            data_yaml_path (str): Path to data.yaml file
            
        Returns:
            Dict: Validation results
        """
        try:
            if self.model is None:
                logger.error("No model available for validation")
                return {}
            
            # Run validation
            val_results = self.model.val(data=data_yaml_path)
            
            validation_metrics = {
                "mAP50": val_results.box.map50 if hasattr(val_results, 'box') else 0.0,
                "mAP50-95": val_results.box.map if hasattr(val_results, 'box') else 0.0,
                "precision": val_results.box.mp if hasattr(val_results, 'box') else 0.0,
                "recall": val_results.box.mr if hasattr(val_results, 'box') else 0.0,
                "validation_results": val_results
            }
            
            logger.info(f"Validation completed: mAP50={validation_metrics['mAP50']:.3f}, "
                       f"mAP50-95={validation_metrics['mAP50-95']:.3f}")
            
            return validation_metrics
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {}


def main():
    """
    Main function for testing the training module.
    """
    # Load configuration
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize trainer
    trainer = YOLOv11Trainer(config)
    
    # Check if data.yaml exists
    data_yaml_path = "data.yaml"
    if not os.path.exists(data_yaml_path):
        print(f"Error: {data_yaml_path} not found. Please run data preprocessing first.")
        return
    
    # Start training
    print("Starting drone detection model training...")
    if trainer.train(data_yaml_path):
        print("Training completed successfully!")
        
        # Get results
        results = trainer.get_training_results()
        if results:
            print(f"Model saved to: {results['model_path']}")
            print(f"Best fitness: {results['best_fitness']:.3f}")
        
        # Validate model
        print("Validating model...")
        val_results = trainer.validate_model(data_yaml_path)
        if val_results:
            print(f"Validation mAP50: {val_results['mAP50']:.3f}")
            print(f"Validation mAP50-95: {val_results['mAP50-95']:.3f}")
    else:
        print("Training failed!")


if __name__ == "__main__":
    main()
