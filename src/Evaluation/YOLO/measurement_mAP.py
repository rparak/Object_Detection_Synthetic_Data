# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../..')
# Matplotlib (Visualization) [pip3 install matplotlib]
import matplotlib.pyplot as plt
# OS (Operating system interfaces)
import os
# SciencePlots (Matplotlib styles for scientific plotting) [pip3 install SciencePlots]
import scienceplots
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
#   ../Lib/Utilities/General
import Lib.Utilities.General
#   ../Lib/Utilities/File_IO
import Lib.Utilities.File_IO as File_IO

"""
Description:
    Initialization of constants.
"""
# The ID of the object to be processed.
CONST_OBJECT_ID = 0
# The identification number of the dataset type.
CONST_DATASET_TYPE = 0
# Name of the dataset.
CONST_DATASET_NAME = f'Dataset_Type_{CONST_DATASET_TYPE}_Obj_ID_{CONST_OBJECT_ID}'
# Number of data to be tested.
CONST_NUM_OF_TEST_DATA = 15

# Mean Average Precision (mAP)
# x - Image Idenfication Number (ID)
# y - Score

def main():
    """
    Description:
        ...
    """

    # Locate the path to the project folder
    project_folder = os.getcwd().split('Blender_Synthetic_Data')[0] + 'Blender_Synthetic_Data'

    # Load a pre-trained YOLO model in the *.onnx format.
    model = cv2.dnn.readNet('yolov8s_custom.onnx')

    for i in range(CONST_NUM_OF_TEST_DATA):
        # Load a raw image from a file.
        image_data = cv2.imread(f'{project_folder}/Data/{CONST_DATASET_NAME}/images/test/Image_{(i + 1):05}.png')
        # Load a label (annotation) from a file.
        label_data = File_IO.Load(f'{project_folder}/Data/{CONST_DATASET_NAME}/labels/test/Image_{(i + 1):05}', 'txt', ' ')

if __name__ == '__main__':
    sys.exit(main())