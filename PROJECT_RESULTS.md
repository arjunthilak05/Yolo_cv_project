# Car Parts Detection - YOLO v11 Results Summary

| Metric              | Value      | Percentage |
| ------------------- | ---------- | ---------- |
| **mAP@[IoU=50]**    | **0.6805** | **68.05%** |
| **mAP@[IoU=50-95]** | **0.5255** | **52.55%** |
| Precision           | 0.622      | 62.2%      |
| Recall              | 0.787      | 78.7%      |

**Training**: 82 epochs, 4.21 hours

## Best Performing Classes

- Front glass (86.07%)
- Front bumper (83.01%)
- Back bumper (78.82%)
- Hood (76.65%)
- Tailgate (72.00%)

## Worst Performing Classes

- Left mirror (33.35%) - **Most difficult**
- Wheel (34.65%)
- Front left light (35.15%)
- Back left light (35.94%)
- Back right door (36.23%)

### Why Left Mirror is Most Difficult

- Small object size compared to bumpers/hood
- Visual similarity between left/right mirrors
- Frequent occlusion and viewing angle variations
- Reflective surfaces create detection challenges
- High IoU precision required for good scores

## Key Improvement Solutions

- **Data**: Increase resolution to 1024×1024, add mirror-specific augmentation

| Model   | Architecture | Resolution | Augmentation   |
| ------- | ------------ | ---------- | -------------- |
| Model 1 | YOLOv11n     | 640        | Standard       |
| Model 2 | YOLOv11s     | 800        | Heavy          |
| Model 3 | YOLOv11m     | 1024       | Moderate       |
| Model 4 | YOLOv11n     | 640        | Mirror-focused |
| Model 5 | YOLOv11s     | 640        | TTA            |

- **Training**: Apply class weighting (2-3x for mirrors), extend to 200 epochs
- **Post-processing**: Lower confidence threshold to 0.15 for mirrors, use test-time augmentation
- **Advanced**: Build ensemble of 3-5 models with Weighted Box Fusion