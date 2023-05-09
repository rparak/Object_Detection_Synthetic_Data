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
#   ../Lib/Utilities/Image_Processing
import Lib.Utilities.Image_Processing

"""
Description:
    Initialization of constants.
"""
# The ID of the object to be processed.
CONST_OBJECT_ID = 1
# The identification number of the dataset type.
CONST_DATASET_TYPE = 0
# Name of the dataset.
CONST_DATASET_NAME = f'Dataset_Type_{CONST_DATASET_TYPE}_Obj_ID_{CONST_OBJECT_ID}'
# Number of data to be tested.
CONST_NUM_OF_TEST_DATA = 15

def main():
    """
    Description:
        ...
    """
        
    # Locate the path to the project folder
    project_folder = os.getcwd().split('Blender_Synthetic_Data')[0] + 'Blender_Synthetic_Data'

    # Load a pre-trained YOLO model in the *.onnx format.
    model = cv2.dnn.readNet(f'{project_folder}/YOLO/Model/Type_{CONST_DATASET_TYPE}_Obj_ID_{CONST_OBJECT_ID}/yolov8s_custom.onnx')

    """
    for i in range(CONST_NUM_OF_TEST_DATA):
        # Load a raw image from a file.
        image_data = cv2.imread(f'{project_folder}/Data/{CONST_DATASET_NAME}/images/test/Image_{(i + 1):05}.png')
    """

if __name__ == '__main__':
    sys.exit(main())