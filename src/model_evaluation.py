"""
Model Evaluation Module for Drone Detection System

This module handles model evaluation, testing, and performance analysis
for the trained drone detection model.
"""

import os
import yaml
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from sklearn.metrics import precision_recall_curve, average_precision_score
import json
from datetime import datetime


class ModelEvaluator:
    """
    Comprehensive model evaluation and testing class.
    """
    
    def __init__(self, model_path: str, config: Dict):
        """
        Initialize model evaluator.
        
        Args:
            model_path (str): Path to the trained model weights
            config (Dict): Configuration dictionary
        """
        self.model_path = model_path
        self.config = config
        self.inference_config = config.get("inference", {})
        self.model = None
        
        # Evaluation metrics storage
        self.evaluation_metrics = {}
        self.detection_results = []
        
    def load_model(self) -> bool:
        """
        Load the trained model.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            self.model = YOLO(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def evaluate_on_dataset(self, data_yaml_path: str, split: str = "val") -> Dict[str, Any]:
        """
        Evaluate model on dataset split.
        
        Args:
            data_yaml_path (str): Path to data.yaml file
            split (str): Dataset split to evaluate on (val, test)
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        try:
            if not self.model:
                if not self.load_model():
                    return {}
            
            # Run evaluation
            logger.info(f"Evaluating model on {split} split...")
            results = self.model.val(
                data=data_yaml_path,
                split=split,
                save=True,
                plots=True,
                conf=self.inference_config.get("conf_threshold", 0.25),
                iou=self.inference_config.get("iou_threshold", 0.45)
            )
            
            # Extract metrics
            evaluation_results = {
                "split": split,
                "mAP50": results.box.map50 if hasattr(results, 'box') else 0.0,
                "mAP50-95": results.box.map if hasattr(results, 'box') else 0.0,
                "precision": results.box.mp if hasattr(results, 'box') else 0.0,
                "recall": results.box.mr if hasattr(results, 'box') else 0.0,
                "f1_score": 2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr) 
                           if (results.box.mp + results.box.mr) > 0 else 0.0,
                "total_detections": len(results) if hasattr(results, '__len__') else 0,
                "results_object": results
            }
            
            # Store results
            self.evaluation_metrics[split] = evaluation_results
            
            logger.info(f"Evaluation completed for {split} split:")
            logger.info(f"  mAP@0.5: {evaluation_results['mAP50']:.3f}")
            logger.info(f"  mAP@0.5:0.95: {evaluation_results['mAP50-95']:.3f}")
            logger.info(f"  Precision: {evaluation_results['precision']:.3f}")
            logger.info(f"  Recall: {evaluation_results['recall']:.3f}")
            logger.info(f"  F1-Score: {evaluation_results['f1_score']:.3f}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Evaluation failed for {split} split: {e}")
            return {}
    
    def test_on_images(self, image_paths: List[str], save_results: bool = True, 
                      output_dir: str = "evaluation_results") -> List[Dict]:
        """
        Test model on individual images.
        
        Args:
            image_paths (List[str]): List of image paths to test
            save_results (bool): Whether to save detection results
            output_dir (str): Directory to save results
            
        Returns:
            List[Dict]: Detection results for each image
        """
        try:
            if not self.model:
                if not self.load_model():
                    return []
            
            if save_results:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
            
            detection_results = []
            
            for i, image_path in enumerate(image_paths):
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}")
                    continue
                
                # Run inference
                results = self.model(
                    image_path,
                    conf=self.inference_config.get("conf_threshold", 0.25),
                    iou=self.inference_config.get("iou_threshold", 0.45),
                    max_det=self.inference_config.get("max_det", 1000)
                )
                
                # Process results
                image_result = {
                    "image_path": image_path,
                    "image_name": os.path.basename(image_path),
                    "detections": [],
                    "total_detections": 0,
                    "confidence_scores": []
                }
                
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        class_ids = result.boxes.cls.cpu().numpy()
                        
                        for box, conf, class_id in zip(boxes, confidences, class_ids):
                            detection = {
                                "bbox": box.tolist(),
                                "confidence": float(conf),
                                "class_id": int(class_id),
                                "class_name": "drone"
                            }
                            image_result["detections"].append(detection)
                            image_result["confidence_scores"].append(float(conf))
                        
                        image_result["total_detections"] = len(image_result["detections"])
                
                detection_results.append(image_result)
                
                # Save annotated image if requested
                if save_results and results:
                    annotated_img = results[0].plot(
                        line_width=self.inference_config.get("line_thickness", 3),
                        font_size=self.inference_config.get("font_size", 1.0),
                        labels=self.inference_config.get("show_labels", True),
                        conf=self.inference_config.get("show_conf", True)
                    )
                    
                    output_filename = f"detected_{os.path.basename(image_path)}"
                    output_filepath = output_path / output_filename
                    cv2.imwrite(str(output_filepath), annotated_img)
                
                logger.info(f"Processed image {i+1}/{len(image_paths)}: {image_result['total_detections']} detections")
            
            self.detection_results = detection_results
            return detection_results
            
        except Exception as e:
            logger.error(f"Image testing failed: {e}")
            return []
    
    def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze model performance across different metrics.
        
        Returns:
            Dict[str, Any]: Performance analysis results
        """
        try:
            if not self.detection_results:
                logger.warning("No detection results available for analysis")
                return {}
            
            # Collect all confidence scores
            all_confidences = []
            detection_counts = []
            
            for result in self.detection_results:
                all_confidences.extend(result["confidence_scores"])
                detection_counts.append(result["total_detections"])
            
            # Calculate statistics
            performance_analysis = {
                "total_images": len(self.detection_results),
                "total_detections": sum(detection_counts),
                "avg_detections_per_image": np.mean(detection_counts),
                "confidence_stats": {
                    "mean": np.mean(all_confidences) if all_confidences else 0.0,
                    "std": np.std(all_confidences) if all_confidences else 0.0,
                    "min": np.min(all_confidences) if all_confidences else 0.0,
                    "max": np.max(all_confidences) if all_confidences else 0.0,
                    "median": np.median(all_confidences) if all_confidences else 0.0
                },
                "detection_distribution": {
                    "images_with_detections": sum(1 for count in detection_counts if count > 0),
                    "images_without_detections": sum(1 for count in detection_counts if count == 0),
                    "max_detections_in_single_image": max(detection_counts) if detection_counts else 0
                }
            }
            
            logger.info("Performance analysis completed:")
            logger.info(f"  Total images: {performance_analysis['total_images']}")
            logger.info(f"  Total detections: {performance_analysis['total_detections']}")
            logger.info(f"  Average detections per image: {performance_analysis['avg_detections_per_image']:.2f}")
            logger.info(f"  Mean confidence: {performance_analysis['confidence_stats']['mean']:.3f}")
            
            return performance_analysis
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {}
    
    def create_evaluation_plots(self, output_dir: str = "evaluation_plots"):
        """
        Create evaluation plots and visualizations.
        
        Args:
            output_dir (str): Directory to save plots
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            if not self.detection_results:
                logger.warning("No detection results available for plotting")
                return
            
            # Collect data for plotting
            all_confidences = []
            detection_counts = []
            
            for result in self.detection_results:
                all_confidences.extend(result["confidence_scores"])
                detection_counts.append(result["total_detections"])
            
            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Drone Detection Model Evaluation', fontsize=16)
            
            # Confidence distribution
            axes[0, 0].hist(all_confidences, bins=30, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 0].set_title('Confidence Score Distribution')
            axes[0, 0].set_xlabel('Confidence Score')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Detection count per image
            axes[0, 1].hist(detection_counts, bins=range(max(detection_counts) + 2), 
                           alpha=0.7, color='green', edgecolor='black')
            axes[0, 1].set_title('Detection Count per Image')
            axes[0, 1].set_xlabel('Number of Detections')
            axes[0, 1].set_ylabel('Number of Images')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Confidence vs Detection Count scatter
            image_confidences = [np.mean(result["confidence_scores"]) if result["confidence_scores"] else 0 
                               for result in self.detection_results]
            axes[1, 0].scatter(detection_counts, image_confidences, alpha=0.6, color='red')
            axes[1, 0].set_title('Detection Count vs Average Confidence')
            axes[1, 0].set_xlabel('Number of Detections')
            axes[1, 0].set_ylabel('Average Confidence')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Performance metrics comparison (if available)
            if self.evaluation_metrics:
                splits = list(self.evaluation_metrics.keys())
                map50_scores = [self.evaluation_metrics[split]['mAP50'] for split in splits]
                map50_95_scores = [self.evaluation_metrics[split]['mAP50-95'] for split in splits]
                
                x = np.arange(len(splits))
                width = 0.35
                
                axes[1, 1].bar(x - width/2, map50_scores, width, label='mAP@0.5', alpha=0.8)
                axes[1, 1].bar(x + width/2, map50_95_scores, width, label='mAP@0.5:0.95', alpha=0.8)
                axes[1, 1].set_title('Model Performance by Dataset Split')
                axes[1, 1].set_xlabel('Dataset Split')
                axes[1, 1].set_ylabel('mAP Score')
                axes[1, 1].set_xticks(x)
                axes[1, 1].set_xticklabels(splits)
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'No evaluation metrics available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Evaluation Metrics')
            
            plt.tight_layout()
            plt.savefig(output_path / 'evaluation_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Evaluation plots saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to create evaluation plots: {e}")
    
    def save_evaluation_report(self, output_file: str = "evaluation_report.json"):
        """
        Save comprehensive evaluation report.
        
        Args:
            output_file (str): Path to save the report
        """
        try:
            report = {
                "evaluation_timestamp": datetime.now().isoformat(),
                "model_path": self.model_path,
                "evaluation_metrics": self.evaluation_metrics,
                "performance_analysis": self.analyze_performance(),
                "detection_summary": {
                    "total_images_processed": len(self.detection_results),
                    "total_detections": sum(result["total_detections"] for result in self.detection_results),
                    "images_with_detections": sum(1 for result in self.detection_results if result["total_detections"] > 0)
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Evaluation report saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save evaluation report: {e}")
    
    def benchmark_inference_speed(self, image_paths: List[str], num_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark model inference speed.
        
        Args:
            image_paths (List[str]): List of image paths for benchmarking
            num_runs (int): Number of runs for averaging
            
        Returns:
            Dict[str, float]: Speed benchmark results
        """
        try:
            if not self.model:
                if not self.load_model():
                    return {}
            
            import time
            
            # Warmup runs
            for _ in range(3):
                for image_path in image_paths[:5]:  # Use first 5 images for warmup
                    self.model(image_path)
            
            # Benchmark runs
            total_times = []
            
            for run in range(num_runs):
                start_time = time.time()
                
                for image_path in image_paths:
                    self.model(image_path)
                
                end_time = time.time()
                total_times.append(end_time - start_time)
            
            # Calculate statistics
            avg_time = np.mean(total_times)
            std_time = np.std(total_times)
            fps = len(image_paths) / avg_time
            
            benchmark_results = {
                "average_inference_time": avg_time,
                "std_inference_time": std_time,
                "fps": fps,
                "time_per_image": avg_time / len(image_paths),
                "num_images": len(image_paths),
                "num_runs": num_runs
            }
            
            logger.info(f"Speed benchmark completed:")
            logger.info(f"  Average time: {avg_time:.3f}s")
            logger.info(f"  FPS: {fps:.2f}")
            logger.info(f"  Time per image: {benchmark_results['time_per_image']:.3f}s")
            
            return benchmark_results
            
        except Exception as e:
            logger.error(f"Speed benchmark failed: {e}")
            return {}


def main():
    """
    Main function for testing the evaluation module.
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
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_path, config)
    
    # Load model
    if not evaluator.load_model():
        print("Failed to load model")
        return
    
    # Evaluate on validation set
    data_yaml_path = "data.yaml"
    if os.path.exists(data_yaml_path):
        print("Evaluating on validation set...")
        val_results = evaluator.evaluate_on_dataset(data_yaml_path, "val")
        
        if val_results:
            print(f"Validation mAP@0.5: {val_results['mAP50']:.3f}")
            print(f"Validation mAP@0.5:0.95: {val_results['mAP50-95']:.3f}")
    
    # Test on sample images (if available)
    test_images_dir = Path("Datasets/drone.v1i.yolov11/test/images")
    if test_images_dir.exists():
        test_images = list(test_images_dir.glob("*.jpg"))[:10]  # Test on first 10 images
        if test_images:
            print(f"Testing on {len(test_images)} sample images...")
            detection_results = evaluator.test_on_images([str(img) for img in test_images])
            
            if detection_results:
                print(f"Processed {len(detection_results)} images")
                
                # Create plots and save report
                evaluator.create_evaluation_plots()
                evaluator.save_evaluation_report()
                print("Evaluation completed successfully!")
    
    print("Model evaluation completed!")


if __name__ == "__main__":
    main()
