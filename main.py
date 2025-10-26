"""
Main Application Entry Point for Drone Detection System

This is the main entry point for the drone detection system using YOLOv11.
It provides both command-line and GUI interfaces for training, evaluation, and inference.
"""

import os
import sys
import yaml
import argparse
import platform
from pathlib import Path
from loguru import logger
from typing import Dict, List, Optional

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))


def check_virtual_environment():
    """
    Check if running inside a virtual environment.
    Provides helpful guidance if not.
    """
    in_venv = (
        hasattr(sys, 'real_prefix') or 
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
        os.environ.get('VIRTUAL_ENV') is not None
    )
    
    if not in_venv:
        print("\n" + "=" * 60)
        print("⚠️  WARNING: Virtual Environment Not Activated!")
        print("=" * 60)
        print("\nYou should run this project in a virtual environment.")
        print("This prevents package conflicts and ensures reproducibility.\n")
        
        print("Quick Setup:")
        print("-" * 60)
        
        if not Path("venv312").exists():
            print("1. Run setup to create virtual environment:")
            if platform.system() == "Windows":
                print("   python setup.py")
                print("   OR: setup.bat\n")
            else:
                print("   python3 setup.py")
                print("   OR: ./setup.sh\n")
        else:
            print("1. Virtual environment exists, just activate it:")
        
        if platform.system() == "Windows":
            print("   .\\venv312\\Scripts\\Activate.ps1")
        else:
            print("   source venv312/bin/activate")
        
        print("\n2. Then run this command again")
        print("=" * 60 + "\n")
        
        response = input("Continue anyway? (not recommended) [y/N]: ").strip().lower()
        if response != 'y':
            print("\nExiting. Please activate the virtual environment first.")
            sys.exit(0)
        else:
            print("\n⚠️  Proceeding without virtual environment (not recommended)\n")

try:
    from data_preprocessing import DatasetManager
    from model_training import YOLOv11Trainer
    from model_evaluation import ModelEvaluator
    from inference import DroneDetector, BatchProcessor
    from gui import DroneDetectionGUI
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)


class DroneDetectionSystem:
    """
    Main class for the drone detection system.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the drone detection system.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.dataset_manager = None
        self.trainer = None
        self.evaluator = None
        self.detector = None
        
        # Setup logging
        self._setup_logging()
    
    def _load_config(self) -> Dict:
        """
        Load configuration from YAML file.
        
        Returns:
            Dict: Configuration dictionary
        """
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return {}
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # Configure loguru
        logger.remove()
        logger.add("logs/drone_detection.log", 
                  rotation="1 day", 
                  retention="7 days",
                  level="INFO")
        logger.add(sys.stderr, level="INFO")
    
    def prepare_dataset(self) -> bool:
        """
        Prepare and validate the dataset.
        
        Returns:
            bool: True if preparation successful, False otherwise
        """
        try:
            logger.info("Starting dataset preparation...")
            
            # Initialize dataset manager
            self.dataset_manager = DatasetManager(self.config)
            
            # Prepare dataset
            if not self.dataset_manager.prepare_dataset():
                logger.error("Dataset preparation failed")
                return False
            
            # Create data.yaml for training
            if not self.dataset_manager.create_data_yaml("data.yaml"):
                logger.error("Failed to create data.yaml file")
                return False
            
            logger.info("Dataset preparation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Dataset preparation failed: {e}")
            return False
    
    def train_model(self) -> bool:
        """
        Train the YOLOv11 model.
        
        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            logger.info("Starting model training...")
            
            # Check if data.yaml exists
            if not os.path.exists("data.yaml"):
                logger.error("data.yaml not found. Please prepare dataset first.")
                return False
            
            # Initialize trainer
            self.trainer = YOLOv11Trainer(self.config)
            
            # Start training
            if not self.trainer.train("data.yaml"):
                logger.error("Model training failed")
                return False
            
            logger.info("Model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False
    
    def evaluate_model(self, model_path: str) -> bool:
        """
        Evaluate the trained model.
        
        Args:
            model_path (str): Path to the trained model
            
        Returns:
            bool: True if evaluation successful, False otherwise
        """
        try:
            logger.info(f"Starting model evaluation: {model_path}")
            
            # Check if model exists
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Initialize evaluator
            self.evaluator = ModelEvaluator(model_path, self.config)
            
            # Load model
            if not self.evaluator.load_model():
                logger.error("Failed to load model for evaluation")
                return False
            
            # Evaluate on validation set
            if os.path.exists("data.yaml"):
                val_results = self.evaluator.evaluate_on_dataset("data.yaml", "val")
                if val_results:
                    logger.info(f"Validation mAP@0.5: {val_results['mAP50']:.3f}")
                    logger.info(f"Validation mAP@0.5:0.95: {val_results['mAP50-95']:.3f}")
            
            # Test on sample images
            test_images_dir = Path("Datasets/drone.v1i.yolov11/test/images")
            if test_images_dir.exists():
                test_images = list(test_images_dir.glob("*.jpg"))[:10]
                if test_images:
                    logger.info(f"Testing on {len(test_images)} sample images...")
                    detection_results = self.evaluator.test_on_images([str(img) for img in test_images])
                    
                    if detection_results:
                        total_detections = sum(result["total_detections"] for result in detection_results)
                        logger.info(f"Total detections: {total_detections}")
                        
                        # Create plots and save report
                        self.evaluator.create_evaluation_plots()
                        self.evaluator.save_evaluation_report()
                        logger.info("Evaluation plots and report saved")
            
            logger.info("Model evaluation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return False
    
    def detect_drones(self, model_path: str, input_path: str, output_path: Optional[str] = None) -> bool:
        """
        Detect drones in images or videos.
        
        Args:
            model_path (str): Path to the trained model
            input_path (str): Path to input image or video
            output_path (Optional[str]): Path to save output
            
        Returns:
            bool: True if detection successful, False otherwise
        """
        try:
            logger.info(f"Starting drone detection: {input_path}")
            
            # Check if model exists
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            # Check if input exists
            if not os.path.exists(input_path):
                logger.error(f"Input file not found: {input_path}")
                return False
            
            # Initialize detector
            self.detector = DroneDetector(model_path, self.config)
            
            # Load model
            if not self.detector.load_model():
                logger.error("Failed to load model for detection")
                return False
            
            # Determine file type and process accordingly
            input_path_obj = Path(input_path)
            file_extension = input_path_obj.suffix.lower()
            
            if file_extension in ['.jpg', '.jpeg', '.png', '.bmp']:
                # Image detection
                result = self.detector.detect_in_image(input_path, output_path)
                if result:
                    logger.info(f"Image detection completed: {result['num_detections']} drones found")
                else:
                    logger.error("Image detection failed")
                    return False
                    
            elif file_extension in ['.mp4', '.avi', '.mov', '.mkv']:
                # Video detection
                result = self.detector.detect_in_video(input_path, output_path)
                if result:
                    logger.info(f"Video detection completed: {result['total_detections']} total detections")
                else:
                    logger.error("Video detection failed")
                    return False
            else:
                logger.error(f"Unsupported file type: {file_extension}")
                return False
            
            logger.info("Drone detection completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Drone detection failed: {e}")
            return False
    
    def run_gui(self):
        """Run the graphical user interface."""
        try:
            logger.info("Starting GUI application...")
            app = DroneDetectionGUI()
            app.run()
        except Exception as e:
            logger.error(f"GUI application failed: {e}")
    
    def get_system_info(self) -> Dict:
        """
        Get system information and status.
        
        Returns:
            Dict: System information
        """
        import torch
        import GPUtil
        import psutil
        
        info = {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count(),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
        }
        
        # GPU information
        if GPUtil.getGPUs():
            gpus = GPUtil.getGPUs()
            info["gpus"] = []
            for gpu in gpus:
                info["gpus"].append({
                    "name": gpu.name,
                    "memory_total": gpu.memoryTotal,
                    "memory_used": gpu.memoryUsed,
                    "temperature": gpu.temperature
                })
        
        return info


def main():
    """Main function for command-line interface."""
    # Check if running in virtual environment
    check_virtual_environment()
    
    parser = argparse.ArgumentParser(description="Drone Detection System using YOLOv11")
    parser.add_argument("--mode", choices=["gui", "train", "evaluate", "detect", "prepare", "info"], 
                       default="gui", help="Operation mode")
    parser.add_argument("--config", default="config/config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--model", help="Path to model file (for evaluate/detect modes)")
    parser.add_argument("--input", help="Path to input file (for detect mode)")
    parser.add_argument("--output", help="Path to output file (for detect mode)")
    parser.add_argument("--skip-venv-check", action="store_true",
                       help="Skip virtual environment check (not recommended)")
    
    args = parser.parse_args()
    
    # Initialize system
    system = DroneDetectionSystem(args.config)
    
    if args.mode == "gui":
        # Run GUI
        system.run_gui()
        
    elif args.mode == "prepare":
        # Prepare dataset
        if system.prepare_dataset():
            print("Dataset preparation completed successfully!")
        else:
            print("Dataset preparation failed!")
            sys.exit(1)
    
    elif args.mode == "train":
        # Train model
        if system.prepare_dataset() and system.train_model():
            print("Model training completed successfully!")
        else:
            print("Model training failed!")
            sys.exit(1)
    
    elif args.mode == "evaluate":
        # Evaluate model
        if not args.model:
            print("Error: --model argument required for evaluate mode")
            sys.exit(1)
        
        if system.evaluate_model(args.model):
            print("Model evaluation completed successfully!")
        else:
            print("Model evaluation failed!")
            sys.exit(1)
    
    elif args.mode == "detect":
        # Detect drones
        if not args.model or not args.input:
            print("Error: --model and --input arguments required for detect mode")
            sys.exit(1)
        
        if system.detect_drones(args.model, args.input, args.output):
            print("Drone detection completed successfully!")
        else:
            print("Drone detection failed!")
            sys.exit(1)
    
    elif args.mode == "info":
        # Show system information
        info = system.get_system_info()
        print("System Information:")
        print(f"Python Version: {info['python_version']}")
        print(f"PyTorch Version: {info['pytorch_version']}")
        print(f"CUDA Available: {info['cuda_available']}")
        if info['cuda_available']:
            print(f"CUDA Version: {info['cuda_version']}")
            print(f"GPU Count: {info['gpu_count']}")
        print(f"CPU Count: {info['cpu_count']}")
        print(f"Memory Total: {info['memory_total'] / (1024**3):.1f} GB")
        print(f"Memory Available: {info['memory_available'] / (1024**3):.1f} GB")
        
        if 'gpus' in info:
            print("\nGPU Information:")
            for i, gpu in enumerate(info['gpus']):
                print(f"  GPU {i}: {gpu['name']}")
                print(f"    Memory: {gpu['memory_used']} / {gpu['memory_total']} MB")
                print(f"    Temperature: {gpu['temperature']}°C")


if __name__ == "__main__":
    main()
