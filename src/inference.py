"""
Inference Module for Drone Detection System

This module handles real-time drone detection using the trained YOLOv11 model
for both image/video files and live camera feed.
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from loguru import logger
import time
from ultralytics import YOLO
import threading
from queue import Queue
import json
from datetime import datetime


class DroneDetector:
    """
    Real-time drone detection using YOLOv11 model.
    """
    
    def __init__(self, model_path: str, config: Dict):
        """
        Initialize drone detector.
        
        Args:
            model_path (str): Path to the trained model weights
            config (Dict): Configuration dictionary
        """
        self.model_path = model_path
        self.config = config
        self.inference_config = config.get("inference", {})
        self.model = None
        
        # Detection parameters
        self.conf_threshold = self.inference_config.get("conf_threshold", 0.25)
        self.iou_threshold = self.inference_config.get("iou_threshold", 0.45)
        self.max_detections = self.inference_config.get("max_det", 1000)
        
        # Visualization parameters
        self.line_thickness = self.inference_config.get("line_thickness", 3)
        self.font_size = self.inference_config.get("font_size", 1.0)
        self.show_labels = self.inference_config.get("show_labels", True)
        self.show_conf = self.inference_config.get("show_conf", True)
        
        # Detection statistics
        self.detection_stats = {
            "total_frames": 0,
            "frames_with_detections": 0,
            "total_detections": 0,
            "avg_confidence": 0.0,
            "fps": 0.0
        }
        
        # Performance monitoring
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
    def load_model(self) -> bool:
        """
        Load the trained YOLOv11 model.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            # Load model
            self.model = YOLO(self.model_path)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Check if GPU is available
            if torch.cuda.is_available():
                logger.info("GPU available for inference")
            else:
                logger.info("Using CPU for inference")
            
            logger.info(f"Drone detection model loaded successfully from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def detect_drones(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect drones in a single image.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            Tuple[np.ndarray, List[Dict]]: Annotated image and detection results
        """
        try:
            if self.model is None:
                logger.error("Model not loaded")
                return image, []
            
            # Run inference
            results = self.model(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                verbose=False
            )
            
            # Process results
            detections = []
            annotated_image = image.copy()
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()
                    
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        # Create detection dictionary
                        detection = {
                            "id": i,
                            "bbox": box.tolist(),
                            "confidence": float(conf),
                            "class_id": int(class_id),
                            "class_name": "drone",
                            "center": [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
                            "width": box[2] - box[0],
                            "height": box[3] - box[1]
                        }
                        detections.append(detection)
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), self.line_thickness)
                        
                        # Draw label and confidence
                        if self.show_labels or self.show_conf:
                            label = ""
                            if self.show_labels:
                                label += "Drone"
                            if self.show_conf:
                                label += f" {conf:.2f}"
                            
                            # Calculate text size
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = self.font_size
                            thickness = max(1, self.line_thickness - 1)
                            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
                            
                            # Draw background rectangle for text
                            cv2.rectangle(annotated_image, 
                                        (x1, y1 - text_height - 10), 
                                        (x1 + text_width, y1), 
                                        (0, 255, 0), -1)
                            
                            # Draw text
                            cv2.putText(annotated_image, label, 
                                      (x1, y1 - 5), font, font_scale, (0, 0, 0), thickness)
            
            # Update statistics
            self._update_stats(len(detections))
            
            return annotated_image, detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return image, []
    
    def _update_stats(self, num_detections: int):
        """
        Update detection statistics.
        
        Args:
            num_detections (int): Number of detections in current frame
        """
        self.detection_stats["total_frames"] += 1
        
        if num_detections > 0:
            self.detection_stats["frames_with_detections"] += 1
            self.detection_stats["total_detections"] += num_detections
        
        # Calculate FPS
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:  # Update FPS every 30 frames
            current_time = time.time()
            elapsed_time = current_time - self.fps_start_time
            self.current_fps = self.fps_counter / elapsed_time
            self.detection_stats["fps"] = self.current_fps
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def detect_in_image(self, image_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect drones in a single image file.
        
        Args:
            image_path (str): Path to input image
            output_path (Optional[str]): Path to save annotated image
            
        Returns:
            Dict[str, Any]: Detection results
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return {}
            
            # Detect drones
            annotated_image, detections = self.detect_drones(image)
            
            # Save annotated image if output path provided
            if output_path:
                cv2.imwrite(output_path, annotated_image)
                logger.info(f"Annotated image saved to: {output_path}")
            
            # Prepare results
            results = {
                "image_path": image_path,
                "image_shape": image.shape,
                "num_detections": len(detections),
                "detections": detections,
                "processing_time": time.time()
            }
            
            logger.info(f"Processed image: {len(detections)} drones detected")
            return results
            
        except Exception as e:
            logger.error(f"Image detection failed: {e}")
            return {}
    
    def detect_in_video(self, video_path: str, output_path: Optional[str] = None, 
                       display: bool = False) -> Dict[str, Any]:
        """
        Detect drones in a video file.
        
        Args:
            video_path (str): Path to input video
            output_path (Optional[str]): Path to save output video
            display (bool): Whether to display video during processing
            
        Returns:
            Dict[str, Any]: Video processing results
        """
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return {}
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Setup video writer if output path provided
            out = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Process video
            frame_count = 0
            total_detections = 0
            start_time = time.time()
            
            logger.info(f"Processing video: {total_frames} frames at {fps} FPS")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect drones in frame
                annotated_frame, detections = self.detect_drones(frame)
                total_detections += len(detections)
                
                # Add FPS counter to frame
                fps_text = f"FPS: {self.current_fps:.1f}"
                cv2.putText(annotated_frame, fps_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add detection count
                det_text = f"Detections: {len(detections)}"
                cv2.putText(annotated_frame, det_text, (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Write frame to output video
                if out:
                    out.write(annotated_frame)
                
                # Display frame if requested
                if display:
                    cv2.imshow('Drone Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
                
                # Progress update
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
            
            # Cleanup
            cap.release()
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()
            
            # Calculate processing statistics
            processing_time = time.time() - start_time
            avg_fps = frame_count / processing_time
            
            results = {
                "video_path": video_path,
                "total_frames": frame_count,
                "total_detections": total_detections,
                "processing_time": processing_time,
                "avg_fps": avg_fps,
                "output_path": output_path
            }
            
            logger.info(f"Video processing completed: {total_detections} total detections in {processing_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Video detection failed: {e}")
            return {}
    
    def detect_live_camera(self, camera_index: int = 0, save_output: bool = False, 
                          output_path: Optional[str] = None) -> None:
        """
        Detect drones in live camera feed.
        
        Args:
            camera_index (int): Camera index (0 for default camera)
            save_output (bool): Whether to save the video stream
            output_path (Optional[str]): Path to save output video
        """
        try:
            # Open camera
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                logger.error(f"Failed to open camera {camera_index}")
                return
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Setup video writer if saving
            out = None
            if save_output and output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            logger.info("Starting live drone detection. Press 'q' to quit, 's' to save screenshot")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read from camera")
                    break
                
                # Detect drones
                annotated_frame, detections = self.detect_drones(frame)
                
                # Add information overlay
                info_text = f"FPS: {self.current_fps:.1f} | Detections: {len(detections)} | Frame: {frame_count}"
                cv2.putText(annotated_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add statistics
                stats_text = f"Total Frames: {self.detection_stats['total_frames']} | "
                stats_text += f"Frames with Detections: {self.detection_stats['frames_with_detections']}"
                cv2.putText(annotated_frame, stats_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Save frame if recording
                if out:
                    out.write(annotated_frame)
                
                # Display frame
                cv2.imshow('Live Drone Detection', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    screenshot_path = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(screenshot_path, annotated_frame)
                    logger.info(f"Screenshot saved: {screenshot_path}")
                
                frame_count += 1
            
            # Cleanup
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            logger.info("Live detection stopped")
            
        except Exception as e:
            logger.error(f"Live detection failed: {e}")
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """
        Get current detection statistics.
        
        Returns:
            Dict[str, Any]: Detection statistics
        """
        return self.detection_stats.copy()
    
    def reset_stats(self):
        """Reset detection statistics."""
        self.detection_stats = {
            "total_frames": 0,
            "frames_with_detections": 0,
            "total_detections": 0,
            "avg_confidence": 0.0,
            "fps": 0.0
        }
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0


class BatchProcessor:
    """
    Batch processing for multiple images/videos.
    """
    
    def __init__(self, detector: DroneDetector):
        """
        Initialize batch processor.
        
        Args:
            detector (DroneDetector): Drone detector instance
        """
        self.detector = detector
        self.results = []
    
    def process_images(self, image_paths: List[str], output_dir: str = "batch_output") -> List[Dict]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths (List[str]): List of image paths
            output_dir (str): Output directory for processed images
            
        Returns:
            List[Dict]: Processing results
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            results = []
            
            for i, image_path in enumerate(image_paths):
                logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
                
                # Generate output filename
                input_filename = Path(image_path).stem
                output_filename = f"{input_filename}_detected.jpg"
                output_filepath = output_path / output_filename
                
                # Process image
                result = self.detector.detect_in_image(image_path, str(output_filepath))
                if result:
                    results.append(result)
            
            self.results = results
            logger.info(f"Batch processing completed: {len(results)} images processed")
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return []
    
    def save_batch_results(self, output_file: str = "batch_results.json"):
        """
        Save batch processing results to JSON file.
        
        Args:
            output_file (str): Path to save results
        """
        try:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Batch results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save batch results: {e}")


def main():
    """
    Main function for testing the inference module.
    """
    # Load configuration
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if model exists
    model_path = "runs/train/drone_detection/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first.")
        return
    
    # Initialize detector
    detector = DroneDetector(model_path, config)
    
    # Load model
    if not detector.load_model():
        print("Failed to load model")
        return
    
    print("Drone detection system ready!")
    print("Available options:")
    print("1. Detect in single image")
    print("2. Detect in video file")
    print("3. Live camera detection")
    print("4. Batch process images")
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        # Single image detection
        image_path = input("Enter image path: ").strip()
        if os.path.exists(image_path):
            result = detector.detect_in_image(image_path, "detected_image.jpg")
            if result:
                print(f"Detection completed: {result['num_detections']} drones found")
        else:
            print("Image file not found")
    
    elif choice == "2":
        # Video detection
        video_path = input("Enter video path: ").strip()
        if os.path.exists(video_path):
            result = detector.detect_in_video(video_path, "detected_video.mp4", display=True)
            if result:
                print(f"Video processing completed: {result['total_detections']} total detections")
        else:
            print("Video file not found")
    
    elif choice == "3":
        # Live camera detection
        camera_index = int(input("Enter camera index (default 0): ") or "0")
        detector.detect_live_camera(camera_index, save_output=True, output_path="live_detection.mp4")
    
    elif choice == "4":
        # Batch processing
        image_dir = input("Enter directory containing images: ").strip()
        if os.path.exists(image_dir):
            image_paths = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
            if image_paths:
                processor = BatchProcessor(detector)
                results = processor.process_images([str(p) for p in image_paths])
                processor.save_batch_results()
                print(f"Batch processing completed: {len(results)} images processed")
            else:
                print("No images found in directory")
        else:
            print("Directory not found")
    
    else:
        print("Invalid choice")


if __name__ == "__main__":
    import yaml
    main()
