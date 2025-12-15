# Quick Start Guide - Car Parts Detection with YOLO v11

## TL;DR - Run the Project in 3 Steps

```bash
# 1. Navigate to project directory
cd /home/user/DL

# 2. Install dependencies (if not already done)
pip install ultralytics pycocotools matplotlib pillow scikit-image numpy jupyter ipykernel

# 3. Run the notebook
jupyter notebook car_parts_yolo_detection.ipynb
```

Then run all cells sequentially from top to bottom.

---

## What This Project Does

This project trains a YOLO v11 deep learning model to detect and locate car parts in images:

- 🚗 Detects 19 different car parts (bumpers, doors, lights, wheels, etc.)
- 📊 Achieves ~60-75% mAP@50 accuracy on test images
- 📸 Provides visual predictions with bounding boxes and confidence scores
- 🔍 Analyzes which car parts are hardest to detect and why
- 💡 Suggests improvements to boost detection accuracy

---

## Project Structure

```
car_parts_yolo_detection.ipynb  ← Main notebook (RUN THIS!)
README.md                       ← Full documentation
QUICKSTART.md                   ← This file
Car-Parts-Segmentation/         ← Dataset (auto-downloaded)
```

---

## Expected Runtime

- **Data preparation**: 2-5 minutes
- **Training**: 30 mins - 2 hours (depends on GPU)
- **Evaluation & analysis**: 2-5 minutes
- **Total**: 40 mins - 2.5 hours

---

## Key Outputs

After running the notebook, you'll get:

1. **Trained Model**: `car_parts_yolo_training/car_parts_exp/weights/best.pt`
2. **Performance Metrics**:
   - mAP@50: Precision at 50% overlap threshold
   - mAP@50-95: Average precision across multiple thresholds
3. **Visualizations**: Prediction images with bounding boxes
4. **Analysis**: Which car parts are difficult to detect and why
5. **Recommendations**: How to improve detection accuracy

---

## Answers to Project Questions

The notebook answers all 4 required questions:

### Q1: What is the mAP@50 and mAP@50-95?
**Answer in**: Section 6 - Model Evaluation
- Calculates both metrics automatically
- Shows per-class performance breakdown

### Q2: Show predictions for 2 random test images
**Answer in**: Section 7 - Prediction Visualization
- Randomly selects 2 test images
- Shows bounding boxes with class names and confidence

### Q3: Which category is difficult to localize?
**Answer in**: Section 8 - Difficult Class Analysis
- Identifies hardest class based on mAP scores
- Analyzes size distribution, data imbalance, etc.
- Explains root causes

### Q4: How to improve detection for difficult class?
**Answer in**: Section 9 - Improvement Solutions
- 7 categories of solutions
- Data augmentation strategies
- Architecture improvements
- Training optimizations

---

## Troubleshooting

### Out of Memory Error
```python
# In Section 5, change batch size:
batch=8  # or batch=4
imgsz=480  # reduce image size
```

### Slow Training
```python
# Use smaller model:
model = YOLO('yolo11n.pt')  # nano version

# Reduce epochs for testing:
epochs=10
```

### No GPU Available
```python
# Force CPU training:
device='cpu'
```

---

## Customization

### Train for Better Accuracy

```python
# Use larger model
model = YOLO('yolo11m.pt')  # medium variant

# Train longer
epochs=200
patience=30

# Higher resolution
imgsz=800
```

### Train Faster (For Testing)

```python
# Smallest model
model = YOLO('yolo11n.pt')

# Fewer epochs
epochs=10

# Smaller images
imgsz=416
batch=32
```

---

## Submission Checklist

Before submitting on Dec 28, 2025:

- [ ] Run entire notebook from top to bottom
- [ ] Verify all 4 questions are answered
- [ ] Check visualizations are generated
- [ ] Export notebook to PDF:
  ```bash
  jupyter nbconvert --to pdf car_parts_yolo_detection.ipynb
  ```
- [ ] Fill in collaboration/AI tools sections
- [ ] Submit both `.ipynb` and `.pdf` files

---

## Need Help?

1. Check `README.md` for detailed documentation
2. Review cell outputs for error messages
3. Consult Ultralytics YOLO docs: https://docs.ultralytics.com/
4. Discuss with classmates (and document it!)

---

**Happy Coding! 🚗📸**

*Estimated completion time: 40 mins - 2.5 hours*
*Submission deadline: 28-Dec-2025, 23:59 Hrs*
