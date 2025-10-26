"""
GUI Module for Drone Detection System

This module provides a comprehensive graphical user interface for
monitoring training, running inference, and managing the drone detection system.
"""

import os
import sys
import yaml
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import numpy as np
from datetime import datetime
import json

# Import our modules
try:
    from .data_preprocessing import DatasetManager
    from .model_training import YOLOv11Trainer, GPUMonitor
    from .model_evaluation import ModelEvaluator
    from .inference import DroneDetector, BatchProcessor
except ImportError:
    # Fallback for direct execution
    from data_preprocessing import DatasetManager
    from model_training import YOLOv11Trainer, GPUMonitor
    from model_evaluation import ModelEvaluator
    from inference import DroneDetector, BatchProcessor


class TrainingMonitorGUI:
    """
    GUI for monitoring training progress and metrics.
    """
    
    def __init__(self, parent_frame: ctk.CTkFrame, config: Dict):
        """
        Initialize training monitor GUI.
        
        Args:
            parent_frame (ctk.CTkFrame): Parent frame for the GUI
            config (Dict): Configuration dictionary
        """
        self.parent_frame = parent_frame
        self.config = config
        self.training_active = False
        self.trainer = None
        self.gpu_monitor = GPUMonitor()
        
        # Create GUI elements
        self._create_training_widgets()
        
    def _create_training_widgets(self):
        """Create training-related GUI widgets."""
        # Training control frame
        control_frame = ctk.CTkFrame(self.parent_frame)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        # Title
        title_label = ctk.CTkLabel(control_frame, text="Model Training", 
                                  font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=10)
        
        # Dataset selection
        dataset_frame = ctk.CTkFrame(control_frame)
        dataset_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(dataset_frame, text="Dataset:").pack(side="left", padx=5)
        self.dataset_var = tk.StringVar(value=self.config.get("dataset", {}).get("primary_dataset", ""))
        self.dataset_entry = ctk.CTkEntry(dataset_frame, textvariable=self.dataset_var, width=300)
        self.dataset_entry.pack(side="left", padx=5)
        
        browse_btn = ctk.CTkButton(dataset_frame, text="Browse", command=self._browse_dataset)
        browse_btn.pack(side="left", padx=5)
        
        # Training parameters
        params_frame = ctk.CTkFrame(control_frame)
        params_frame.pack(fill="x", padx=10, pady=5)
        
        # Epochs
        ctk.CTkLabel(params_frame, text="Epochs:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.epochs_var = tk.StringVar(value=str(self.config.get("model", {}).get("epochs", 100)))
        epochs_entry = ctk.CTkEntry(params_frame, textvariable=self.epochs_var, width=100)
        epochs_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Batch size
        ctk.CTkLabel(params_frame, text="Batch Size:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.batch_size_var = tk.StringVar(value=str(self.config.get("model", {}).get("batch_size", 16)))
        batch_size_entry = ctk.CTkEntry(params_frame, textvariable=self.batch_size_var, width=100)
        batch_size_entry.grid(row=0, column=3, padx=5, pady=5)
        
        # Model size
        ctk.CTkLabel(params_frame, text="Model Size:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.model_size_var = tk.StringVar(value=self.config.get("model", {}).get("size", "m"))
        model_size_combo = ctk.CTkComboBox(params_frame, values=["n", "s", "m", "l", "x"], 
                                          variable=self.model_size_var, width=100)
        model_size_combo.grid(row=1, column=1, padx=5, pady=5)
        
        # Device
        ctk.CTkLabel(params_frame, text="Device:").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        self.device_var = tk.StringVar(value=self.config.get("model", {}).get("device", "0"))
        device_entry = ctk.CTkEntry(params_frame, textvariable=self.device_var, width=100)
        device_entry.grid(row=1, column=3, padx=5, pady=5)
        
        # Control buttons
        button_frame = ctk.CTkFrame(control_frame)
        button_frame.pack(fill="x", padx=10, pady=5)
        
        self.start_btn = ctk.CTkButton(button_frame, text="Start Training", 
                                      command=self._start_training, width=120)
        self.start_btn.pack(side="left", padx=5)
        
        self.stop_btn = ctk.CTkButton(button_frame, text="Stop Training", 
                                     command=self._stop_training, width=120, state="disabled")
        self.stop_btn.pack(side="left", padx=5)
        
        self.validate_btn = ctk.CTkButton(button_frame, text="Validate Dataset", 
                                         command=self._validate_dataset, width=120)
        self.validate_btn.pack(side="left", padx=5)
        
        # Progress frame
        progress_frame = ctk.CTkFrame(control_frame)
        progress_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(progress_frame, text="Training Progress:").pack(anchor="w", padx=5, pady=2)
        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.pack(fill="x", padx=5, pady=2)
        self.progress_bar.set(0)
        
        self.progress_label = ctk.CTkLabel(progress_frame, text="Ready to start training")
        self.progress_label.pack(anchor="w", padx=5, pady=2)
        
        # GPU info frame
        gpu_frame = ctk.CTkFrame(control_frame)
        gpu_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(gpu_frame, text="GPU Information:").pack(anchor="w", padx=5, pady=2)
        self.gpu_info_text = ctk.CTkTextbox(gpu_frame, height=100)
        self.gpu_info_text.pack(fill="x", padx=5, pady=2)
        
        # Update GPU info
        self._update_gpu_info()
        
    def _browse_dataset(self):
        """Browse for dataset directory."""
        dataset_path = filedialog.askdirectory(title="Select Dataset Directory")
        if dataset_path:
            self.dataset_var.set(dataset_path)
    
    def _update_gpu_info(self):
        """Update GPU information display."""
        try:
            gpu_info = self.gpu_monitor.get_gpu_info()
            
            if gpu_info["available"]:
                info_text = f"GPU Available: {gpu_info['count']} device(s)\n\n"
                for device in gpu_info["devices"]:
                    info_text += f"Device {device['id']}: {device['name']}\n"
                    info_text += f"  Memory: {device['memory_used']} / {device['memory_total']}\n"
                    info_text += f"  Utilization: {device['gpu_utilization']}\n"
                    info_text += f"  Temperature: {device['temperature']}\n\n"
                
                # Get recommended batch size
                recommended_batch = self.gpu_monitor.get_recommended_batch_size(
                    self.model_size_var.get()
                )
                info_text += f"Recommended Batch Size: {recommended_batch}\n"
            else:
                info_text = "No GPU detected. Training will use CPU (slower).\n"
                info_text += "Recommended Batch Size: 8"
            
            self.gpu_info_text.delete("1.0", "end")
            self.gpu_info_text.insert("1.0", info_text)
            
        except Exception as e:
            self.gpu_info_text.delete("1.0", "end")
            self.gpu_info_text.insert("1.0", f"Error getting GPU info: {e}")
    
    def _validate_dataset(self):
        """Validate selected dataset."""
        dataset_path = self.dataset_var.get()
        if not dataset_path:
            messagebox.showerror("Error", "Please select a dataset directory")
            return
        
        try:
            # Update progress
            self.progress_label.configure(text="Validating dataset...")
            self.parent_frame.update()
            
            # Validate dataset
            dataset_manager = DatasetManager(self.config)
            dataset_manager.primary_dataset = dataset_path
            
            if dataset_manager.prepare_dataset():
                messagebox.showinfo("Success", "Dataset validation passed!")
                self.progress_label.configure(text="Dataset validation completed")
            else:
                messagebox.showerror("Error", "Dataset validation failed!")
                self.progress_label.configure(text="Dataset validation failed")
                
        except Exception as e:
            messagebox.showerror("Error", f"Dataset validation error: {e}")
            self.progress_label.configure(text="Dataset validation error")
    
    def _start_training(self):
        """Start model training."""
        dataset_path = self.dataset_var.get()
        if not dataset_path:
            messagebox.showerror("Error", "Please select a dataset directory")
            return
        
        try:
            # Update config with current values
            self.config["dataset"]["primary_dataset"] = dataset_path
            self.config["model"]["epochs"] = int(self.epochs_var.get())
            self.config["model"]["batch_size"] = int(self.batch_size_var.get())
            self.config["model"]["size"] = self.model_size_var.get()
            self.config["model"]["device"] = self.device_var.get()
            
            # Initialize trainer
            self.trainer = YOLOv11Trainer(self.config)
            
            # Create data.yaml
            data_yaml_path = "data.yaml"
            dataset_manager = DatasetManager(self.config)
            if not dataset_manager.create_data_yaml(data_yaml_path):
                messagebox.showerror("Error", "Failed to create data.yaml file")
                return
            
            # Start training in separate thread
            self.training_active = True
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            
            training_thread = threading.Thread(target=self._training_worker, args=(data_yaml_path,))
            training_thread.daemon = True
            training_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start training: {e}")
            self._reset_training_buttons()
    
    def _training_worker(self, data_yaml_path: str):
        """Training worker thread."""
        try:
            self.progress_label.configure(text="Starting training...")
            
            # Start training
            success = self.trainer.train(data_yaml_path)
            
            if success:
                self.progress_label.configure(text="Training completed successfully!")
                messagebox.showinfo("Success", "Training completed successfully!")
            else:
                self.progress_label.configure(text="Training failed!")
                messagebox.showerror("Error", "Training failed!")
                
        except Exception as e:
            self.progress_label.configure(text=f"Training error: {e}")
            messagebox.showerror("Error", f"Training error: {e}")
        finally:
            self._reset_training_buttons()
    
    def _stop_training(self):
        """Stop model training."""
        self.training_active = False
        self.progress_label.configure(text="Stopping training...")
        # Note: YOLOv11 doesn't have built-in stop functionality
        # This would need to be implemented with process management
        self._reset_training_buttons()
    
    def _reset_training_buttons(self):
        """Reset training control buttons."""
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.training_active = False


class InferenceGUI:
    """
    GUI for running inference and detection.
    """
    
    def __init__(self, parent_frame: ctk.CTkFrame, config: Dict):
        """
        Initialize inference GUI.
        
        Args:
            parent_frame (ctk.CTkFrame): Parent frame for the GUI
            config (Dict): Configuration dictionary
        """
        self.parent_frame = parent_frame
        self.config = config
        self.detector = None
        self.live_detection_active = False
        
        # Create GUI elements
        self._create_inference_widgets()
    
    def _create_inference_widgets(self):
        """Create inference-related GUI widgets."""
        # Inference control frame
        control_frame = ctk.CTkFrame(self.parent_frame)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        # Title
        title_label = ctk.CTkLabel(control_frame, text="Drone Detection", 
                                  font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=10)
        
        # Model selection
        model_frame = ctk.CTkFrame(control_frame)
        model_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(model_frame, text="Model Path:").pack(side="left", padx=5)
        self.model_path_var = tk.StringVar(value="runs/train/drone_detection/weights/best.pt")
        self.model_path_entry = ctk.CTkEntry(model_frame, textvariable=self.model_path_var, width=300)
        self.model_path_entry.pack(side="left", padx=5)
        
        browse_model_btn = ctk.CTkButton(model_frame, text="Browse", command=self._browse_model)
        browse_model_btn.pack(side="left", padx=5)
        
        load_model_btn = ctk.CTkButton(model_frame, text="Load Model", command=self._load_model)
        load_model_btn.pack(side="left", padx=5)
        
        # Detection parameters
        params_frame = ctk.CTkFrame(control_frame)
        params_frame.pack(fill="x", padx=10, pady=5)
        
        # Confidence threshold
        ctk.CTkLabel(params_frame, text="Confidence:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.conf_var = tk.DoubleVar(value=self.config.get("inference", {}).get("conf_threshold", 0.25))
        conf_slider = ctk.CTkSlider(params_frame, from_=0.0, to=1.0, variable=self.conf_var, width=150)
        conf_slider.grid(row=0, column=1, padx=5, pady=5)
        self.conf_label = ctk.CTkLabel(params_frame, text=f"{self.conf_var.get():.2f}")
        self.conf_label.grid(row=0, column=2, padx=5, pady=5)
        conf_slider.configure(command=self._update_conf_label)
        
        # IoU threshold
        ctk.CTkLabel(params_frame, text="IoU Threshold:").grid(row=0, column=3, padx=5, pady=5, sticky="w")
        self.iou_var = tk.DoubleVar(value=self.config.get("inference", {}).get("iou_threshold", 0.45))
        iou_slider = ctk.CTkSlider(params_frame, from_=0.0, to=1.0, variable=self.iou_var, width=150)
        iou_slider.grid(row=0, column=4, padx=5, pady=5)
        self.iou_label = ctk.CTkLabel(params_frame, text=f"{self.iou_var.get():.2f}")
        self.iou_label.grid(row=0, column=5, padx=5, pady=5)
        iou_slider.configure(command=self._update_iou_label)
        
        # Detection modes
        mode_frame = ctk.CTkFrame(control_frame)
        mode_frame.pack(fill="x", padx=10, pady=5)
        
        # Single image detection
        image_frame = ctk.CTkFrame(mode_frame)
        image_frame.pack(side="left", fill="x", expand=True, padx=5)
        
        ctk.CTkLabel(image_frame, text="Single Image Detection").pack(pady=5)
        self.image_path_var = tk.StringVar()
        image_entry = ctk.CTkEntry(image_frame, textvariable=self.image_path_var, width=200)
        image_entry.pack(side="left", padx=5)
        
        browse_image_btn = ctk.CTkButton(image_frame, text="Browse", command=self._browse_image)
        browse_image_btn.pack(side="left", padx=5)
        
        detect_image_btn = ctk.CTkButton(image_frame, text="Detect", command=self._detect_image)
        detect_image_btn.pack(side="left", padx=5)
        
        # Video detection
        video_frame = ctk.CTkFrame(mode_frame)
        video_frame.pack(side="left", fill="x", expand=True, padx=5)
        
        ctk.CTkLabel(video_frame, text="Video Detection").pack(pady=5)
        self.video_path_var = tk.StringVar()
        video_entry = ctk.CTkEntry(video_frame, textvariable=self.video_path_var, width=200)
        video_entry.pack(side="left", padx=5)
        
        browse_video_btn = ctk.CTkButton(video_frame, text="Browse", command=self._browse_video)
        browse_video_btn.pack(side="left", padx=5)
        
        detect_video_btn = ctk.CTkButton(video_frame, text="Detect", command=self._detect_video)
        detect_video_btn.pack(side="left", padx=5)
        
        # Live detection
        live_frame = ctk.CTkFrame(control_frame)
        live_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(live_frame, text="Live Camera Detection").pack(pady=5)
        
        camera_frame = ctk.CTkFrame(live_frame)
        camera_frame.pack(fill="x", padx=5, pady=5)
        
        ctk.CTkLabel(camera_frame, text="Camera Index:").pack(side="left", padx=5)
        self.camera_var = tk.StringVar(value="0")
        camera_entry = ctk.CTkEntry(camera_frame, textvariable=self.camera_var, width=50)
        camera_entry.pack(side="left", padx=5)
        
        self.start_live_btn = ctk.CTkButton(camera_frame, text="Start Live Detection", 
                                           command=self._start_live_detection)
        self.start_live_btn.pack(side="left", padx=5)
        
        self.stop_live_btn = ctk.CTkButton(camera_frame, text="Stop Live Detection", 
                                          command=self._stop_live_detection, state="disabled")
        self.stop_live_btn.pack(side="left", padx=5)
        
        # Detection results
        results_frame = ctk.CTkFrame(control_frame)
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        ctk.CTkLabel(results_frame, text="Detection Results").pack(pady=5)
        self.results_text = ctk.CTkTextbox(results_frame, height=200)
        self.results_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Detection statistics
        stats_frame = ctk.CTkFrame(control_frame)
        stats_frame.pack(fill="x", padx=10, pady=5)
        
        self.stats_label = ctk.CTkLabel(stats_frame, text="No detections yet")
        self.stats_label.pack(pady=5)
    
    def _update_conf_label(self, value):
        """Update confidence threshold label."""
        self.conf_label.configure(text=f"{float(value):.2f}")
    
    def _update_iou_label(self, value):
        """Update IoU threshold label."""
        self.iou_label.configure(text=f"{float(value):.2f}")
    
    def _browse_model(self):
        """Browse for model file."""
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        if model_path:
            self.model_path_var.set(model_path)
    
    def _browse_image(self):
        """Browse for image file."""
        image_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if image_path:
            self.image_path_var.set(image_path)
    
    def _browse_video(self):
        """Browse for video file."""
        video_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if video_path:
            self.video_path_var.set(video_path)
    
    def _load_model(self):
        """Load the detection model."""
        model_path = self.model_path_var.get()
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("Error", "Please select a valid model file")
            return
        
        try:
            # Update config
            self.config["inference"]["conf_threshold"] = self.conf_var.get()
            self.config["inference"]["iou_threshold"] = self.iou_var.get()
            
            # Initialize detector
            self.detector = DroneDetector(model_path, self.config)
            
            if self.detector.load_model():
                self.results_text.insert("end", f"Model loaded successfully: {model_path}\n")
                messagebox.showinfo("Success", "Model loaded successfully!")
            else:
                messagebox.showerror("Error", "Failed to load model")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
    
    def _detect_image(self):
        """Detect drones in single image."""
        if not self.detector:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        image_path = self.image_path_var.get()
        if not image_path or not os.path.exists(image_path):
            messagebox.showerror("Error", "Please select a valid image file")
            return
        
        try:
            # Update detector parameters
            self.detector.conf_threshold = self.conf_var.get()
            self.detector.iou_threshold = self.iou_var.get()
            
            # Run detection
            output_path = f"detected_{os.path.basename(image_path)}"
            result = self.detector.detect_in_image(image_path, output_path)
            
            if result:
                self.results_text.insert("end", f"Image: {os.path.basename(image_path)}\n")
                self.results_text.insert("end", f"Detections: {result['num_detections']}\n")
                self.results_text.insert("end", f"Output saved: {output_path}\n\n")
                
                # Update statistics
                stats = self.detector.get_detection_stats()
                stats_text = f"Total Frames: {stats['total_frames']} | "
                stats_text += f"Frames with Detections: {stats['frames_with_detections']} | "
                stats_text += f"Total Detections: {stats['total_detections']}"
                self.stats_label.configure(text=stats_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {e}")
    
    def _detect_video(self):
        """Detect drones in video file."""
        if not self.detector:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        video_path = self.video_path_var.get()
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Error", "Please select a valid video file")
            return
        
        try:
            # Update detector parameters
            self.detector.conf_threshold = self.conf_var.get()
            self.detector.iou_threshold = self.iou_var.get()
            
            # Run detection
            output_path = f"detected_{os.path.basename(video_path)}"
            result = self.detector.detect_in_video(video_path, output_path, display=True)
            
            if result:
                self.results_text.insert("end", f"Video: {os.path.basename(video_path)}\n")
                self.results_text.insert("end", f"Total Detections: {result['total_detections']}\n")
                self.results_text.insert("end", f"Processing Time: {result['processing_time']:.2f}s\n")
                self.results_text.insert("end", f"Average FPS: {result['avg_fps']:.2f}\n")
                self.results_text.insert("end", f"Output saved: {output_path}\n\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Video detection failed: {e}")
    
    def _start_live_detection(self):
        """Start live camera detection."""
        if not self.detector:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        try:
            camera_index = int(self.camera_var.get())
            
            # Update detector parameters
            self.detector.conf_threshold = self.conf_var.get()
            self.detector.iou_threshold = self.iou_var.get()
            
            # Start live detection in separate thread
            self.live_detection_active = True
            self.start_live_btn.configure(state="disabled")
            self.stop_live_btn.configure(state="normal")
            
            live_thread = threading.Thread(target=self._live_detection_worker, args=(camera_index,))
            live_thread.daemon = True
            live_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start live detection: {e}")
    
    def _live_detection_worker(self, camera_index: int):
        """Live detection worker thread."""
        try:
            self.detector.detect_live_camera(camera_index, save_output=True, 
                                           output_path="live_detection.mp4")
        except Exception as e:
            self.results_text.insert("end", f"Live detection error: {e}\n")
        finally:
            self._reset_live_buttons()
    
    def _stop_live_detection(self):
        """Stop live camera detection."""
        self.live_detection_active = False
        # Note: This would need to be implemented with proper thread management
        self._reset_live_buttons()
    
    def _reset_live_buttons(self):
        """Reset live detection buttons."""
        self.start_live_btn.configure(state="normal")
        self.stop_live_btn.configure(state="disabled")
        self.live_detection_active = False


class DroneDetectionGUI:
    """
    Main GUI application for the drone detection system.
    """
    
    def __init__(self):
        """Initialize the main GUI application."""
        # Set appearance mode and color theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Create main window
        self.root = ctk.CTk()
        self.root.title("Drone Detection System")
        self.root.geometry("1200x800")
        
        # Load configuration
        self.config = self._load_config()
        
        # Create GUI elements
        self._create_main_widgets()
        
        # Setup logging
        self._setup_logging()
    
    def _load_config(self) -> Dict:
        """Load configuration from file."""
        try:
            with open("config/config.yaml", 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # Configure loguru
        logger.remove()
        logger.add("logs/drone_detection.log", rotation="1 day", retention="7 days")
        logger.add(sys.stderr, level="INFO")
    
    def _create_main_widgets(self):
        """Create main GUI widgets."""
        # Title
        title_label = ctk.CTkLabel(self.root, text="Drone Detection System", 
                                  font=ctk.CTkFont(size=28, weight="bold"))
        title_label.pack(pady=20)
        
        # Create notebook for tabs
        self.notebook = ctk.CTkTabview(self.root)
        self.notebook.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Training tab
        self.training_tab = self.notebook.add("Training")
        self.training_monitor = TrainingMonitorGUI(self.training_tab, self.config)
        
        # Inference tab
        self.inference_tab = self.notebook.add("Detection")
        self.inference_gui = InferenceGUI(self.inference_tab, self.config)
        
        # Evaluation tab
        self.evaluation_tab = self.notebook.add("Evaluation")
        self._create_evaluation_widgets()
        
        # Settings tab
        self.settings_tab = self.notebook.add("Settings")
        self._create_settings_widgets()
        
        # Status bar
        self.status_frame = ctk.CTkFrame(self.root)
        self.status_frame.pack(fill="x", padx=20, pady=5)
        
        self.status_label = ctk.CTkLabel(self.status_frame, text="Ready")
        self.status_label.pack(side="left", padx=10, pady=5)
        
        # Version info
        version_label = ctk.CTkLabel(self.status_frame, text="v1.0.0")
        version_label.pack(side="right", padx=10, pady=5)
    
    def _create_evaluation_widgets(self):
        """Create evaluation tab widgets."""
        # Evaluation control frame
        control_frame = ctk.CTkFrame(self.evaluation_tab)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(control_frame, text="Model Evaluation", 
                    font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)
        
        # Model selection
        model_frame = ctk.CTkFrame(control_frame)
        model_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(model_frame, text="Model Path:").pack(side="left", padx=5)
        self.eval_model_var = tk.StringVar(value="runs/train/drone_detection/weights/best.pt")
        eval_model_entry = ctk.CTkEntry(model_frame, textvariable=self.eval_model_var, width=300)
        eval_model_entry.pack(side="left", padx=5)
        
        eval_browse_btn = ctk.CTkButton(model_frame, text="Browse", 
                                       command=lambda: self._browse_file(self.eval_model_var, "*.pt"))
        eval_browse_btn.pack(side="left", padx=5)
        
        # Evaluation buttons
        button_frame = ctk.CTkFrame(control_frame)
        button_frame.pack(fill="x", padx=10, pady=5)
        
        eval_btn = ctk.CTkButton(button_frame, text="Evaluate Model", 
                                command=self._evaluate_model)
        eval_btn.pack(side="left", padx=5)
        
        test_images_btn = ctk.CTkButton(button_frame, text="Test on Images", 
                                       command=self._test_on_images)
        test_images_btn.pack(side="left", padx=5)
        
        # Results display
        results_frame = ctk.CTkFrame(self.evaluation_tab)
        results_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        ctk.CTkLabel(results_frame, text="Evaluation Results").pack(pady=5)
        self.eval_results_text = ctk.CTkTextbox(results_frame)
        self.eval_results_text.pack(fill="both", expand=True, padx=5, pady=5)
    
    def _create_settings_widgets(self):
        """Create settings tab widgets."""
        # Settings control frame
        control_frame = ctk.CTkFrame(self.settings_tab)
        control_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(control_frame, text="System Settings", 
                    font=ctk.CTkFont(size=20, weight="bold")).pack(pady=10)
        
        # Configuration display
        config_frame = ctk.CTkFrame(control_frame)
        config_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        ctk.CTkLabel(config_frame, text="Current Configuration").pack(pady=5)
        self.config_text = ctk.CTkTextbox(config_frame)
        self.config_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Load current config
        self._load_config_display()
    
    def _browse_file(self, var: tk.StringVar, file_type: str):
        """Browse for file and update variable."""
        file_path = filedialog.askopenfilename(filetypes=[(file_type, file_type)])
        if file_path:
            var.set(file_path)
    
    def _evaluate_model(self):
        """Evaluate the selected model."""
        model_path = self.eval_model_var.get()
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("Error", "Please select a valid model file")
            return
        
        try:
            evaluator = ModelEvaluator(model_path, self.config)
            
            if not evaluator.load_model():
                messagebox.showerror("Error", "Failed to load model")
                return
            
            # Evaluate on validation set
            data_yaml_path = "data.yaml"
            if os.path.exists(data_yaml_path):
                self.eval_results_text.insert("end", "Evaluating on validation set...\n")
                val_results = evaluator.evaluate_on_dataset(data_yaml_path, "val")
                
                if val_results:
                    self.eval_results_text.insert("end", f"mAP@0.5: {val_results['mAP50']:.3f}\n")
                    self.eval_results_text.insert("end", f"mAP@0.5:0.95: {val_results['mAP50-95']:.3f}\n")
                    self.eval_results_text.insert("end", f"Precision: {val_results['precision']:.3f}\n")
                    self.eval_results_text.insert("end", f"Recall: {val_results['recall']:.3f}\n")
                    self.eval_results_text.insert("end", f"F1-Score: {val_results['f1_score']:.3f}\n\n")
            else:
                self.eval_results_text.insert("end", "data.yaml not found. Please train a model first.\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Evaluation failed: {e}")
    
    def _test_on_images(self):
        """Test model on sample images."""
        model_path = self.eval_model_var.get()
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("Error", "Please select a valid model file")
            return
        
        try:
            evaluator = ModelEvaluator(model_path, self.config)
            
            if not evaluator.load_model():
                messagebox.showerror("Error", "Failed to load model")
                return
            
            # Test on sample images
            test_images_dir = Path("Datasets/drone.v1i.yolov11/test/images")
            if test_images_dir.exists():
                test_images = list(test_images_dir.glob("*.jpg"))[:10]
                if test_images:
                    self.eval_results_text.insert("end", f"Testing on {len(test_images)} sample images...\n")
                    detection_results = evaluator.test_on_images([str(img) for img in test_images])
                    
                    if detection_results:
                        total_detections = sum(result["total_detections"] for result in detection_results)
                        self.eval_results_text.insert("end", f"Total detections: {total_detections}\n")
                        self.eval_results_text.insert("end", f"Images processed: {len(detection_results)}\n")
                        
                        # Create plots and save report
                        evaluator.create_evaluation_plots()
                        evaluator.save_evaluation_report()
                        self.eval_results_text.insert("end", "Evaluation plots and report saved.\n")
                else:
                    self.eval_results_text.insert("end", "No test images found.\n")
            else:
                self.eval_results_text.insert("end", "Test images directory not found.\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Image testing failed: {e}")
    
    def _load_config_display(self):
        """Load and display current configuration."""
        try:
            config_text = yaml.dump(self.config, default_flow_style=False, indent=2)
            self.config_text.insert("1.0", config_text)
        except Exception as e:
            self.config_text.insert("1.0", f"Error loading config: {e}")
    
    def run(self):
        """Run the GUI application."""
        self.root.mainloop()


def main():
    """Main function to run the GUI application."""
    try:
        app = DroneDetectionGUI()
        app.run()
    except Exception as e:
        logger.error(f"GUI application failed: {e}")
        messagebox.showerror("Error", f"Application failed to start: {e}")


if __name__ == "__main__":
    main()
