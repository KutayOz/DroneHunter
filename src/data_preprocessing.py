"""
Data Preprocessing Module for Drone Detection System

This module handles dataset preparation, validation, and augmentation
for the drone detection system using YOLOv11 format.
"""

import os
import yaml
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from loguru import logger
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DatasetValidator:
    """
    Validates YOLOv11 dataset structure and annotations.
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize dataset validator.
        
        Args:
            dataset_path (str): Path to the dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self.data_yaml_path = self.dataset_path / "data.yaml"
        
    def validate_structure(self) -> bool:
        """
        Validate dataset directory structure.
        
        Returns:
            bool: True if structure is valid, False otherwise
        """
        required_dirs = ["train/images", "train/labels", "valid/images", "valid/labels"]
        required_files = ["data.yaml"]
        
        # Check required directories
        for dir_name in required_dirs:
            dir_path = self.dataset_path / dir_name
            if not dir_path.exists():
                logger.error(f"Missing required directory: {dir_path}")
                return False
                
        # Check required files
        for file_name in required_files:
            file_path = self.dataset_path / file_name
            if not file_path.exists():
                logger.error(f"Missing required file: {file_path}")
                return False
                
        logger.info("Dataset structure validation passed")
        return True
    
    def validate_annotations(self) -> Tuple[int, int, List[str]]:
        """
        Validate YOLO format annotations.
        
        Returns:
            Tuple[int, int, List[str]]: (valid_annotations, invalid_annotations, error_messages)
        """
        valid_count = 0
        invalid_count = 0
        error_messages = []
        
        # Check training annotations
        train_labels_dir = self.dataset_path / "train/labels"
        for label_file in train_labels_dir.glob("*.txt"):
            if not self._validate_single_annotation(label_file):
                invalid_count += 1
                error_messages.append(f"Invalid annotation: {label_file}")
            else:
                valid_count += 1
                
        # Check validation annotations
        valid_labels_dir = self.dataset_path / "valid/labels"
        for label_file in valid_labels_dir.glob("*.txt"):
            if not self._validate_single_annotation(label_file):
                invalid_count += 1
                error_messages.append(f"Invalid annotation: {label_file}")
            else:
                valid_count += 1
                
        logger.info(f"Annotation validation: {valid_count} valid, {invalid_count} invalid")
        return valid_count, invalid_count, error_messages
    
    def _validate_single_annotation(self, label_file: Path) -> bool:
        """
        Validate a single annotation file.
        
        Args:
            label_file (Path): Path to the annotation file
            
        Returns:
            bool: True if annotation is valid, False otherwise
        """
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) != 5:
                    return False
                    
                # Check if all values are numeric and within valid ranges
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Validate ranges
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                           0 <= width <= 1 and 0 <= height <= 1):
                        return False
                        
                except ValueError:
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error validating annotation {label_file}: {e}")
            return False
    
    def get_dataset_info(self) -> Dict:
        """
        Get comprehensive dataset information.
        
        Returns:
            Dict: Dataset information including counts and statistics
        """
        info = {
            "dataset_path": str(self.dataset_path),
            "train_images": len(list((self.dataset_path / "train/images").glob("*"))),
            "train_labels": len(list((self.dataset_path / "train/labels").glob("*.txt"))),
            "valid_images": len(list((self.dataset_path / "valid/images").glob("*"))),
            "valid_labels": len(list((self.dataset_path / "valid/labels").glob("*.txt"))),
        }
        
        # Add test set info if exists
        test_images_dir = self.dataset_path / "test/images"
        if test_images_dir.exists():
            info["test_images"] = len(list(test_images_dir.glob("*")))
            info["test_labels"] = len(list((self.dataset_path / "test/labels").glob("*.txt")))
        
        return info


class DataAugmentation:
    """
    Handles data augmentation for drone detection training.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize data augmentation with configuration.
        
        Args:
            config (Dict): Augmentation configuration from config.yaml
        """
        self.config = config
        self.augmentation_pipeline = self._create_augmentation_pipeline()
    
    def _create_augmentation_pipeline(self) -> A.Compose:
        """
        Create augmentation pipeline based on configuration.
        
        Returns:
            A.Compose: Albumentations augmentation pipeline
        """
        if not self.config.get("enabled", True):
            return A.Compose([ToTensorV2()])
        
        transforms = []
        
        # Color space augmentations
        if self.config.get("hsv_h", 0) > 0 or self.config.get("hsv_s", 0) > 0 or self.config.get("hsv_v", 0) > 0:
            transforms.append(A.HueSaturationValue(
                hue_shift_limit=int(self.config.get("hsv_h", 0) * 180),
                sat_shift_limit=int(self.config.get("hsv_s", 0) * 255),
                val_shift_limit=int(self.config.get("hsv_v", 0) * 255),
                p=0.5
            ))
        
        # Geometric augmentations
        if self.config.get("degrees", 0) > 0:
            transforms.append(A.Rotate(
                limit=int(self.config.get("degrees", 0)),
                p=0.5
            ))
        
        if self.config.get("translate", 0) > 0:
            translate_limit = self.config.get("translate", 0)
            transforms.append(A.ShiftScaleRotate(
                shift_limit=translate_limit,
                scale_limit=0.1,
                rotate_limit=0,
                p=0.5
            ))
        
        # Flip augmentations
        if self.config.get("fliplr", 0) > 0:
            transforms.append(A.HorizontalFlip(p=self.config.get("fliplr", 0)))
        
        if self.config.get("flipud", 0) > 0:
            transforms.append(A.VerticalFlip(p=self.config.get("flipud", 0)))
        
        # Brightness and contrast
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ))
        
        # Noise and blur
        transforms.append(A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.MotionBlur(blur_limit=3, p=0.3),
        ], p=0.3))
        
        # Weather effects
        transforms.append(A.OneOf([
            A.RandomRain(p=0.1),
            A.RandomSnow(p=0.1),
            A.RandomSunFlare(p=0.1),
        ], p=0.2))
        
        transforms.append(ToTensorV2())
        
        return A.Compose(transforms, bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels']
        ))
    
    def augment_image(self, image: np.ndarray, bboxes: List[List], class_labels: List[int]) -> Tuple[np.ndarray, List[List], List[int]]:
        """
        Apply augmentation to image and bounding boxes.
        
        Args:
            image (np.ndarray): Input image
            bboxes (List[List]): Bounding boxes in YOLO format
            class_labels (List[int]): Class labels for each bounding box
            
        Returns:
            Tuple[np.ndarray, List[List], List[int]]: Augmented image, bboxes, and labels
        """
        try:
            augmented = self.augmentation_pipeline(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            return augmented['image'], augmented['bboxes'], augmented['class_labels']
        except Exception as e:
            logger.warning(f"Augmentation failed, returning original: {e}")
            return image, bboxes, class_labels


class DatasetManager:
    """
    Manages dataset operations including preparation and validation.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize dataset manager.
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config
        self.dataset_config = config.get("dataset", {})
        self.primary_dataset = self.dataset_config.get("primary_dataset", "")
        
    def prepare_dataset(self) -> bool:
        """
        Prepare and validate the primary dataset.
        
        Returns:
            bool: True if preparation successful, False otherwise
        """
        if not self.primary_dataset:
            logger.error("No primary dataset specified in configuration")
            return False
        
        # Validate dataset structure
        validator = DatasetValidator(self.primary_dataset)
        if not validator.validate_structure():
            logger.error("Dataset structure validation failed")
            return False
        
        # Validate annotations
        valid_count, invalid_count, errors = validator.validate_annotations()
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} invalid annotations")
            for error in errors[:10]:  # Show first 10 errors
                logger.warning(error)
        
        # Get dataset information
        dataset_info = validator.get_dataset_info()
        logger.info(f"Dataset prepared successfully: {dataset_info}")
        
        return True
    
    def get_dataset_path(self) -> str:
        """
        Get the path to the primary dataset.
        
        Returns:
            str: Path to the dataset
        """
        return self.primary_dataset
    
    def create_data_yaml(self, output_path: str) -> bool:
        """
        Create a properly formatted data.yaml file for YOLOv11 training.
        
        Args:
            output_path (str): Path where to save the data.yaml file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Read original data.yaml
            original_yaml_path = Path(self.primary_dataset) / "data.yaml"
            with open(original_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            
            # Update paths to be absolute
            dataset_path = Path(self.primary_dataset).resolve()
            data_config['train'] = str(dataset_path / "train/images")
            data_config['val'] = str(dataset_path / "valid/images")
            data_config['test'] = str(dataset_path / "test/images")
            
            # Ensure class names are consistent
            data_config['names'] = ['drone']  # Ensure drone is the only class
            data_config['nc'] = 1  # Number of classes
            
            # Save updated data.yaml
            with open(output_path, 'w') as f:
                yaml.dump(data_config, f, default_flow_style=False)
            
            logger.info(f"Created data.yaml at {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create data.yaml: {e}")
            return False


def main():
    """
    Main function for testing the data preprocessing module.
    """
    # Load configuration
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize dataset manager
    dataset_manager = DatasetManager(config)
    
    # Prepare dataset
    if dataset_manager.prepare_dataset():
        print("Dataset preparation completed successfully!")
        
        # Create data.yaml for training
        dataset_manager.create_data_yaml("data.yaml")
    else:
        print("Dataset preparation failed!")


if __name__ == "__main__":
    main()
