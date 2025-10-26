# Dataset Setup Guide

## 📦 Important Notice

**Datasets are NOT included in this repository** due to their large size. You must download and prepare your own drone detection dataset before training.

## 🔍 Where to Get Datasets

### Option 1: Roboflow Universe
- Visit: https://universe.roboflow.com/search?q=drone
- Search for "drone detection" datasets
- Download in YOLOv11 format
- Free datasets available with account

### Option 2: Kaggle
- Visit: https://www.kaggle.com/search?q=drone+detection
- Browse drone detection datasets
- Download and extract
- Convert to YOLO format if needed

### Option 3: Custom Dataset
- Collect your own drone images
- Annotate using tools like:
  - [LabelImg](https://github.com/tzutalin/labelImg)
  - [CVAT](https://github.com/opencv/cvat)
  - [Roboflow](https://roboflow.com/)
- Export in YOLO format

## 📁 Required Directory Structure

After downloading, organize your dataset like this:

```
Datasets/
├── your-dataset-name/
│   ├── train/
│   │   ├── images/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   └── ...
│   │   └── labels/
│   │       ├── image1.txt
│   │       ├── image2.txt
│   │       └── ...
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
```

## 🔧 Setup Steps

### 1. Create Datasets Directory

```bash
mkdir Datasets
cd Datasets
```

### 2. Download Dataset

Download your chosen dataset and extract it into the `Datasets/` folder.

### 3. Verify Structure

Ensure your dataset follows the structure above with:
- `train/` folder with images and labels
- `valid/` folder with images and labels
- `test/` folder with images and labels (optional)

### 4. Update Configuration

Edit `config/config.yaml` and set the dataset path:

```yaml
dataset:
  primary_dataset: "Datasets/your-dataset-name"
```

### 5. Prepare Dataset

Run the preparation script:

```bash
python main.py --mode prepare
```

This will:
- Validate dataset structure
- Check for missing files
- Create `data.yaml` configuration
- Display dataset statistics

## 📊 Dataset Requirements

### Minimum Requirements
- **Training images**: 100+ (1000+ recommended)
- **Validation images**: 20+ (200+ recommended)
- **Test images**: 10+ (100+ recommended)
- **Image format**: JPG, PNG, or BMP
- **Label format**: YOLO format (class x_center y_center width height)

### Recommended Dataset Size
- **Small dataset**: 500-1000 images
- **Medium dataset**: 1000-5000 images
- **Large dataset**: 5000+ images

## 🏷️ YOLO Label Format

Each `.txt` label file should contain one line per object:

```
class_id x_center y_center width height
```

Where:
- `class_id`: 0 for drone (or other class IDs)
- `x_center`, `y_center`: Center coordinates (normalized 0-1)
- `width`, `height`: Bounding box dimensions (normalized 0-1)

Example:
```
0 0.5 0.5 0.3 0.2
```

## ✅ Verification

After setup, verify your dataset:

```bash
python main.py --mode info
```

Check the logs for:
- Number of training images
- Number of validation images
- Class distribution
- Any warnings or errors

## 🚨 Common Issues

### Issue: "Dataset not found"
**Solution**: Check the path in `config/config.yaml` matches your actual dataset location.

### Issue: "Missing labels"
**Solution**: Ensure every image has a corresponding `.txt` file with the same name.

### Issue: "Empty labels"
**Solution**: Some images may not have annotations. This is okay, but ensure most images are labeled.

### Issue: "Invalid label format"
**Solution**: Check that labels follow YOLO format with normalized coordinates (0-1).

## 📝 Example Datasets

### Recommended Datasets for Drone Detection

1. **Drone Detection Dataset v1**
   - Size: ~20,000 images
   - Source: Roboflow Universe
   - Classes: drone

2. **UAV Detection Dataset**
   - Size: ~10,000 images
   - Source: Kaggle
   - Classes: drone, bird

3. **Anti-Drone Dataset**
   - Size: ~5,000 images
   - Source: Academic research
   - Classes: various drone types

## 🔄 Data Augmentation

The system automatically applies augmentation during training:
- Horizontal flip
- HSV color space adjustments
- Mosaic augmentation
- Translation and scaling

Configure in `config/config.yaml`:

```yaml
dataset:
  augmentation:
    enabled: true
    fliplr: 0.5
    hsv_h: 0.015
    hsv_s: 0.7
    hsv_v: 0.4
```

## 💡 Tips

1. **Balance your dataset**: Try to have similar numbers of images in each class
2. **Diverse conditions**: Include images with different:
   - Lighting conditions
   - Weather conditions
   - Backgrounds
   - Drone sizes and distances
3. **Quality over quantity**: Better to have 1000 well-labeled images than 10,000 poorly labeled ones
4. **Test set**: Keep a separate test set that the model never sees during training

## 📞 Need Help?

If you encounter issues:
1. Check the logs in `logs/drone_detection.log`
2. Verify dataset structure matches the required format
3. Run dataset validation: `python main.py --mode prepare`
4. Create an issue on GitHub with details

---

**Remember**: The model is only as good as the data you train it on! 🎯

