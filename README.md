# Tooth-and-Cavity-Instance-Segmentation
## Overview
This project aims to develop a deep learning model for instance segmentation of teeth, cavities, caries, and cracks, from dental images. The model leverages a Mask R-CNN architecture with a ResNet50 backbone to achieve high accuracy and fidelity in identifying and segmenting these structures.

## Features
- Instance Segmentation: Accurately segments and identifies multiple instances of dental structures.
- Model: Utilizes a MASK RCNN model (with PyTorch's default weights) for enhanced performance.
- TensorBoard for visualizing and monitoring training
- Technologies Used:
  - Python
  - PyTorch
  - Torchvision
  - OpenCV
  - NumPy
  - Matplotlib
  - Jupyter Notebook

## Dataset
The dataset used for training and validation is sourced from the Arab Academy on Roboflow. It includes bounding boxes, masks, and segmentations in the COCO format.
- https://universe.roboflow.com/arab-academy-vf9su/dental-7yegp

## Model Training
- Model Architecture: Mask R-CNN with ResNet50 backbone.
- Training Parameters:
  - Initial Learning Rate: 0.001
  - Max Epochs: 50
  - Batch Size: 2 (Training/Validation), 8 (Testing)
  - Early Stop Patience: 5
  - Learning Rate Scheduler: ReduceLROnPlateau with factor 0.1 and patience 3
- Performance
  - Mean Average Precision (mAP): 0.79
  - Precision: 0.742
  - Recall: 0.815
  - F1-Score: 0.777

## Installation / Usage
1. Clone the repository:
```sh
git clone https://github.com/benjamin-ahlbrecht/Tooth-and-Cavity-Instance-Segmentation.git
cd Tooth-and-Cavity-Instance-Segmentation
```

2. Install the required dependencies:
```
# Create and activate a Python virtual environment [Optional]
python -m venv .venv
source .venv/Scripts/activate

# Install requirements
pip install -r requirements.txt
```

3. Launch the Jupyter server
```sh
cd notebooks
jupyter-lab .
```

4. Run and monitor the code. `train_model` is set to `False` by default. If you want to train or validate the model yourself, please ensure you have ample VRAM (~4 GB / Image Instance). Reducing/tuning the batch size may be required. Implementing gradient accumulation (or similar method) to reduce VRAM usage during training may be helpful.

## Visualization
This project includes functionality to visualize bounding boxes, masks, and segmentations of each observation using Matplotlib.
