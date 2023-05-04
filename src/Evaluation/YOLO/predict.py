# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../..')
# OS (Operating system interfaces)
import os
# Time (Time access and conversions)
import time
# OpenCV (Computer Vision) [pip3 install opencv-python]
import cv2
# PyTorch (Tensors and Dynamic neural networks) [pip3 install torch]
import torch
# Torchvision (Image and video datasets and models) [pip3 install torchvision]
import torchvision.ops.boxes
# Custom Library:
#   ../Lib/Utilities/General
import Lib.Utilities.General

def main():
    # Locate the path to the project folder
    project_folder = os.getcwd().split('Blender_Synthetic_Data')[0] + 'Blender_Synthetic_Data'

if __name__ == '__main__':
    sys.exit(main())