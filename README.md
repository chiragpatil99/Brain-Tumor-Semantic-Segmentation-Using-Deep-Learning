# Brain Tumor Semantic Segmentation Using Deep Learning

## Overview
This project focuses on the semantic segmentation of brain tumors using deep learning techniques. We implemented and evaluated various models, including U-Net and YOLOv9, for segmenting brain tumors from multimodal magnetic resonance imaging (MRI) scans. The study used the BraTS2020 dataset for training, validation, and testing.

## Authors
- **Chirag Patil** - Base U-Net Model and Modified U-Net with Attention Mechanism Implementation
- **Pranav Pawar** - Dice Score Calculation
- **Shubham** - YOLOv9 Implementation and Transfer Learning

## Dataset
The dataset used for this project is the **BraTS2020** (Multimodal Brain Tumor Segmentation Challenge) dataset, which contains MRI scans with the following modalities:
- FLAIR
- T1
- T1ce
- T2

The dataset includes corresponding segmentation masks for different tumor regions:
- Enhancing Tumor
- Tumor Core
- Surrounding Edema

You can access the dataset from the following links:
- [Kaggle - BraTS20 Dataset](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)
- [CBICA BraTS2020 Data](https://www.med.upenn.edu/cbica/brats2020/data.html)

## Data Processing Pipeline
- **Loading and Normalizing MRI Data:** Normalization using MinMaxScaler to scale voxel values between 0 and 1.
- **Loading and Reassigning Segmentation Masks:** Converting segmentation masks to appropriate labels.
- **Visualization:** MRI images and segmentation masks are visualized using Matplotlib for inspection.
- **Stacking and Cropping:** 3D volumes are cropped to (128x128x128) for model input.
- **Saving Processed Data:** Processed data is saved in `.npy` format.
- **Data Splitting:** Dataset split into 75% training and 25% validation using `splitfolders`.

## Model Architecture
The primary architecture used in this study is **U-Net** with the following components:
- **Contracting Path:** Convolutional and MaxPooling layers for feature extraction.
- **Bottleneck:** Deep convolutional layers to extract complex features.
- **Expansive Path:** Transposed convolutions for upsampling and precise localization.
- **Skip Connections:** Retaining high-resolution features for accurate segmentation.

Additionally, we experimented with a **Modified U-Net with Attention Mechanism** and **YOLOv9** using transfer learning.

## Experimental Results
Metrics used for evaluation:
- **Loss:** Measures model error during training and validation.
- **Accuracy:** Overall percentage of correct predictions.
- **Intersection over Union (IoU) Score:** Evaluates segmentation overlap with ground truth.

### Results Summary
- **U-Net:** Achieved satisfactory results with further room for optimization.
- **YOLOv9 with Transfer Learning:** Improved accuracy and detection.
- **Modified U-Net with Attention:** Enhanced segmentation accuracy with better feature focus.

## Conclusion
While U-Net was effective in segmenting brain tumors, it faced challenges related to memory demands and overfitting. Transfer learning using YOLOv9 showed improvements in detection accuracy. The attention mechanism further enhanced the model’s segmentation performance.

Future research directions include improving model generalization, reducing overfitting, and integrating additional clinical data sources.


## References
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
- BraTS Dataset and Challenge: [CBICA BraTS2020 Data](https://www.med.upenn.edu/cbica/brats2020/data.html)

---

For any questions or contributions, feel free to contact **Chirag Patil** at chiragpatil@vt.edu.

**Happy Coding!**

