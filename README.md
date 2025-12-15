# Car Parts Detection using YOLO v11

## Computer Vision Course Project (CDS402)

This project implements an object detection system for car parts using YOLO v11 (You Only Look Once) on the Car Parts Segmentation dataset.

---

## 📋 Project Overview

### Objectives
1. Train a YOLO v11 model on the Car Parts Dataset
2. Calculate mAP@50 and mAP@50-95 metrics on the test dataset
3. Visualize predictions with bounding boxes, class names, and confidence scores
4. Identify the most difficult category to localize
5. Propose solutions to improve detection accuracy for difficult classes

### Dataset Information
- **Source**: [Car Parts Segmentation Dataset](https://github.com/dsmlr/Car-Parts-Segmentation)
- **Training images**: 400
- **Test images**: 100
- **Annotation format**: COCO format (JSON)
- **Number of classes**: 19 car part categories

### Classes
1. \_background\_
2. back_bumper
3. back_glass
4. back_left_door
5. back_left_light
6. back_right_door
7. back_right_light
8. front_bumper
9. front_glass
10. front_left_door
11. front_left_light
12. front_right_door
13. front_right_light
14. hood
15. left_mirror
16. right_mirror
17. tailgate
18. trunk
19. wheel

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+ (tested with Python 3.11.14)
- CUDA-compatible GPU (recommended for faster training)
- 10+ GB free disk space
- Internet connection (for downloading packages and pretrained weights)

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd /home/user/DL
   ```

2. **Install required dependencies:**
   ```bash
   pip install ultralytics pycocotools matplotlib pillow scikit-image numpy jupyter ipykernel
   ```

3. **Verify installation:**
   ```bash
   python3 -c "import ultralytics; import pycocotools; print('All packages installed successfully!')"
   ```

---

## 📓 Running the Project

### Option 1: Using Jupyter Notebook (Recommended)

1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open the notebook:**
   - Navigate to `car_parts_yolo_detection.ipynb`
   - Run cells sequentially from top to bottom

3. **Execution time:**
   - Data preparation: 2-5 minutes
   - Training (100 epochs): 30 minutes - 2 hours (depending on GPU)
   - Evaluation and visualization: 2-5 minutes

### Option 2: Using Jupyter Lab

```bash
jupyter lab car_parts_yolo_detection.ipynb
```

### Option 3: Command-line execution

Convert notebook to Python script and run:
```bash
jupyter nbconvert --to script car_parts_yolo_detection.ipynb
python3 car_parts_yolo_detection.py
```

---

## 📊 Project Structure

```
DL/
├── Car-Parts-Segmentation/        # Original dataset (cloned from GitHub)
│   ├── trainingset/
│   │   ├── JPEGImages/            # Training images
│   │   └── annotations.json       # COCO format annotations
│   └── testset/
│       ├── JPEGImages/            # Test images
│       └── annotations.json       # COCO format annotations
│
├── car_parts_yolo/                # YOLO format dataset (created by notebook)
│   ├── images/
│   │   ├── train/                 # Training images
│   │   └── val/                   # Validation/test images
│   ├── labels/
│   │   ├── train/                 # Training labels (YOLO format)
│   │   └── val/                   # Validation labels (YOLO format)
│   └── data.yaml                  # YOLO dataset configuration
│
├── car_parts_yolo_training/       # Training outputs
│   └── car_parts_exp/
│       ├── weights/
│       │   ├── best.pt            # Best model weights
│       │   └── last.pt            # Last epoch weights
│       ├── results.csv            # Training metrics
│       └── results.png            # Training curves
│
├── car_parts_yolo_detection.ipynb # Main project notebook
└── README.md                      # This file
```

---

## 🔧 Notebook Workflow

The notebook is organized into the following sections:

### 1. Import Libraries
Imports all necessary packages including ultralytics, pycocotools, matplotlib, etc.

### 2. Data Exploration
- Loads and explores the COCO format annotations
- Displays class distribution
- Shows dataset statistics

### 3. COCO to YOLO Conversion
- Converts bounding box format from COCO to YOLO
- Creates proper directory structure
- Generates label files

### 4. Dataset Configuration
- Creates YAML configuration file
- Defines paths and class names

### 5. Model Training
- Loads pretrained YOLO v11 model
- Trains on car parts dataset
- Saves best weights

### 6. Model Evaluation
- **Answers Question 1**: Calculates mAP@50 and mAP@50-95
- Evaluates on test set
- Generates per-class metrics

### 7. Prediction Visualization
- **Answers Question 2**: Shows predictions on 2 random test images
- Displays bounding boxes with class names and confidence

### 8. Difficult Class Analysis
- **Answers Question 3**: Identifies most difficult class to localize
- Analyzes reasons (data imbalance, object size, etc.)
- Provides detailed statistics

### 9. Improvement Solutions
- **Answers Question 4**: Proposes comprehensive solutions
- Suggests data augmentation techniques
- Recommends architectural improvements

### 10. Results Summary
- Consolidates all findings
- Displays final metrics

---

## 📈 Expected Results

### Performance Metrics
- **mAP@50**: Typically 0.60-0.75 (60-75%)
- **mAP@50-95**: Typically 0.35-0.50 (35-50%)

Results vary depending on:
- Training epochs
- Model size (nano vs small vs medium)
- GPU availability
- Hyperparameter settings

### Output Files
1. **Trained model**: `car_parts_yolo_training/car_parts_exp/weights/best.pt`
2. **Training curves**: `car_parts_yolo_training/car_parts_exp/results.png`
3. **Prediction visualizations**: `prediction_*.jpg`
4. **Metrics**: `car_parts_yolo_training/car_parts_exp/results.csv`

---

## ❓ Project Questions & Answers

### Question 1: What is the mAP@[IoU=50] and mAP@[IoU=50-95] for the testing dataset?

**Location in Notebook**: Section 6 - "Evaluate Model and Calculate mAP Metrics"

The notebook calculates:
- **mAP@50**: Mean Average Precision at IoU threshold of 0.5
- **mAP@50-95**: Mean Average Precision averaged over IoU thresholds from 0.5 to 0.95 (COCO metric)
- **Per-class mAP**: Individual mAP scores for each of the 19 car part categories

### Question 2: Show visualization of predictions for two random samples

**Location in Notebook**: Section 7 - "Visualize Predictions on Test Samples"

The notebook:
- Randomly selects 2 test images
- Runs inference with the trained model
- Displays predictions with:
  - Bounding boxes (red rectangles)
  - Class names
  - Confidence scores
- Saves visualizations as PNG files

### Question 3: Which category is difficult to correctly localize?

**Location in Notebook**: Section 8 - "Analyze Difficult-to-Localize Categories"

The notebook implements a comprehensive strategy:
1. **mAP-based ranking**: Ranks classes by mAP score (lowest = most difficult)
2. **Class distribution analysis**: Examines training sample counts
3. **Object size analysis**: Calculates average bbox area and variance
4. **Visual analysis**: Plots size distributions
5. **Root cause identification**: Determines why the class is difficult

**Possible reasons analyzed:**
- Data imbalance (insufficient training samples)
- Small object size (hard to detect at low resolution)
- High size variance (inconsistent appearance)
- Visual similarity to other parts
- Occlusion rates

### Question 4: How to improve detection accuracy for the difficult class?

**Location in Notebook**: Section 9 - "Solutions to Improve Detection Accuracy"

The notebook proposes **7 comprehensive solution categories**:

1. **Data Augmentation**:
   - Copy-paste augmentation
   - Class-specific transformations
   - Synthetic data generation

2. **Model Architecture**:
   - Use larger YOLO variants
   - Multi-scale detection
   - Attention mechanisms

3. **Training Strategy**:
   - Class weighting
   - Focal loss
   - Two-stage training

4. **Input Resolution**:
   - Increase resolution for small objects
   - Adaptive resolution training

5. **Post-processing**:
   - Class-specific confidence thresholds
   - Adjusted NMS parameters

6. **Ensemble Methods**:
   - Multiple model architectures
   - Weighted Box Fusion

7. **Dataset Quality**:
   - Annotation review
   - Diverse sample collection

---

## ⚙️ Customization

### Training Hyperparameters

Modify in Section 5 of the notebook:

```python
results = model.train(
    data=yaml_path,
    epochs=100,        # Increase for better performance (e.g., 200-300)
    imgsz=640,        # Increase for small objects (e.g., 800, 1024)
    batch=16,         # Adjust based on GPU memory
    patience=20,      # Early stopping patience
    device='cuda',    # Use 'cpu' if no GPU available
)
```

### Model Selection

Change YOLO variant for different speed/accuracy tradeoffs:

```python
# Nano (fastest, lowest accuracy)
model = YOLO('yolo11n.pt')

# Small (balanced)
model = YOLO('yolo11s.pt')

# Medium (better accuracy, slower)
model = YOLO('yolo11m.pt')

# Large/Extra Large (best accuracy, slowest)
model = YOLO('yolo11l.pt')
model = YOLO('yolo11x.pt')
```

---

## 🐛 Troubleshooting

### Common Issues

**1. CUDA out of memory**
- Reduce batch size: `batch=8` or `batch=4`
- Reduce image size: `imgsz=480`
- Use CPU: `device='cpu'`

**2. Installation errors**
```bash
# Install packages individually
pip install torch torchvision
pip install ultralytics
pip install pycocotools
```

**3. Slow training**
- Ensure GPU is being used (check `device='cuda'`)
- Reduce epochs for testing: `epochs=10`
- Use smaller model: `yolo11n.pt`

**4. Module not found**
```bash
# Reinstall in current kernel
pip install --force-reinstall ultralytics pycocotools
```

---

## 📝 Submission Checklist

Before submitting, ensure you have:

- [✓] Completed notebook (`.ipynb` file)
- [✓] Exported PDF of notebook
- [✓] Answered all 4 questions in the notebook
- [✓] Generated prediction visualizations
- [✓] Calculated mAP metrics
- [✓] Analyzed difficult classes
- [✓] Proposed improvement solutions
- [✓] Listed AI tools used (if any)
- [✓] Listed classmates discussed with (if any)

### Exporting to PDF

**Option 1: Using Jupyter**
```bash
jupyter nbconvert --to pdf car_parts_yolo_detection.ipynb
```

**Option 2: Using nbconvert**
```bash
pip install nbconvert[webpdf]
jupyter nbconvert --to webpdf car_parts_yolo_detection.ipynb
```

**Option 3: Print from browser**
- Open notebook in Jupyter
- File → Print Preview
- Print to PDF

---

## 📚 References

1. **Ultralytics YOLOv11 Documentation**
   https://docs.ultralytics.com/

2. **COCO Dataset Format**
   https://cocodataset.org/#format-data

3. **Mean Average Precision (mAP)**
   https://www.ultralytics.com/glossary/mean-average-precision-map

4. **Car Parts Segmentation Dataset**
   https://github.com/dsmlr/Car-Parts-Segmentation

5. **YOLO Original Paper** (Redmon et al.)
   https://arxiv.org/abs/1506.02640

---

## 👥 Collaboration & AI Tools

### AI Tools Used
This project was developed with assistance from AI tools (Claude) for:
- Understanding YOLO v11 API and best practices
- COCO to YOLO format conversion logic
- Structuring analysis and visualization code
- Formulating comprehensive improvement suggestions

All code was reviewed, understood, and tested before inclusion. The analysis and conclusions are based on actual model results.

### Classmates Discussed With
*(Fill in after discussing with classmates)*

Example:
- **Student Name**: Topic discussed and how it helped
- **Student Name**: Topic discussed and how it helped

---

## ⏱ Estimated Timeline

| Task | Estimated Time |
|------|----------------|
| Setup & Installation | 10-15 minutes |
| Data Exploration | 5-10 minutes |
| Format Conversion | 5 minutes |
| Model Training | 30 min - 2 hours |
| Evaluation | 5-10 minutes |
| Analysis & Documentation | 15-30 minutes |
| **Total** | **1-3.5 hours** |

*Note: Training time varies significantly based on hardware*

---

## 📧 Contact & Support

For questions or issues:
1. Review the notebook comments and documentation
2. Check Ultralytics documentation: https://docs.ultralytics.com/
3. Consult course materials
4. Discuss with classmates (and document in notebook)

---

## 📄 License

This project uses:
- **YOLO v11**: AGPL-3.0 License (Ultralytics)
- **Car Parts Dataset**: Check original repository for licensing
- **Project code**: Educational use for CDS402 course

---

## ✅ Submission Deadline

**Date**: 28-Dec-2025
**Time**: 23:59 Hrs

**Submit**:
1. Notebook file (`car_parts_yolo_detection.ipynb`)
2. PDF export of notebook
3. Any additional visualizations or results

---

**Good luck with your project! 🚗🔍**
