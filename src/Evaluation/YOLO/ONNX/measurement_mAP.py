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

"""
Description:
    Initialization of constants.
"""
# The ID of the object to be tested.
CONST_OBJECT_ID = 0
# Available objects.
#   ID{0} = 'T_Joint'
#   ID{1} = 'Metal_Blank'
CONST_OBJECT_NAME = ['T_Joint', 'Metal_Blank']
# The identification number of the dataset type.
CONST_DATASET_TYPE = 0
# Name of the dataset.
CONST_DATASET_NAME = f'Dataset_Type_{CONST_DATASET_TYPE}_Obj_ID_{CONST_OBJECT_ID}'
# Number of data to be tested.
CONST_NUM_OF_TEST_DATA = 15
# Iteration of the testing process.
CONST_SCAN_ITERATION = 30

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
    model = cv2.dnn.readNet(f'{project_folder}/YOLO/Model/Type_{CONST_DATASET_TYPE}_Obj_ID_{CONST_OBJECT_ID}/yolov8s_custom.onnx')

    for n_i in range(CONST_NUM_OF_TEST_DATA):
        # Loads images from the specified file.
        image_data = cv2.imread(f'{project_folder}/Data/{CONST_DATASET_NAME}/images/test/Image_{(CONST_SCAN_ITERATION + (n_i + 1)):05}.png')

        # Object detection using the trained YOLO model.
        (class_id, bounding_box, confidence) = Lib.Utilities.Image_Processing.YOLO_Object_Detection(image_data, model, 640, 0.5)

        if class_id != None:
            for _, (class_id_i, bounding_box_i, confidence_i) in enumerate(zip(class_id, bounding_box, confidence)):
                print(confidence_i)

if __name__ == '__main__':
    sys.exit(main())