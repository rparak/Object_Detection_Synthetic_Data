# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../../..')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
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
#   ../Lib/Utilities/File_IO
import Lib.Utilities.File_IO as File_IO
#   ../Lib/Transformation/Utilities/Mathematics
import Lib.Transformation.Utilities.Mathematics as Mathematics

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
CONST_DATASET_NAME = f'Dataset_Type_{CONST_DATASET_TYPE}'
# Number of data to be tested.
CONST_NUM_OF_TEST_DATA = 1
# Initial iteration of the scanning process.
CONST_SCAN_ITERATION = 30

# Mean Average Precision (mAP)
# x - Image Idenfication Number (ID)
# y - Score

# Score:
#   Intersection over Union (IoU)
#   Confidence

def main():
    """
    Description:
        ...
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Blender_Synthetic_Data')[0] + 'Blender_Synthetic_Data'

    # Load a pre-trained YOLO model in the *.onnx format.
    model = cv2.dnn.readNet(f'{project_folder}/YOLO/Model/Type_{CONST_DATASET_TYPE}/yolov8n_custom.onnx')

    score_confidence = []; score_iou = []; total_num_of_data = 0; num_of_data = []
    for n_i in range(CONST_NUM_OF_TEST_DATA):
        image_file_path = f'{project_folder}/Data/{CONST_DATASET_NAME}/images/test/Object_ID_{CONST_OBJECT_ID}_{(CONST_SCAN_ITERATION + (n_i + 1)):05}.png'

        # Loads images from the specified file.
        image_data = cv2.imread(image_file_path)

        # Loads labels (annotations) from the specified file.
        label_data = File_IO.Load(f'{project_folder}/Data/{CONST_DATASET_NAME}/labels/test/Object_ID_{CONST_OBJECT_ID}_{(CONST_SCAN_ITERATION + (n_i + 1)):05}', 'txt', ' ')

        # ...
        bounding_box_desired = []
        for _, label_data_i in enumerate(label_data):
            bounding_box_desired_tmp = Lib.Utilities.General.Convert_Boundig_Box_Data('YOLO', 'PASCAL_VOC', 
                                                                                      {'x_c': label_data_i[1], 
                                                                                       'y_c': label_data_i[2], 
                                                                                       'width': label_data_i[3], 
                                                                                       'height': label_data_i[4]}, {'x': image_data.shape[1], 'y': image_data.shape[0]})
            bounding_box_desired.append(list(bounding_box_desired_tmp.values()))

        # Calculate the number of data in the current row and the total number of data.
        total_num_of_data += len(bounding_box_desired)
        num_of_data.append(len(bounding_box_desired))
        
        # Object detection using the trained YOLO model.
        # ..... REWRITE IMAGE SIZE !!!!!!!
        (class_id_predicted_tmp, bounding_box_predicted_tmp, confidence_predicted_tmp) = Lib.Utilities.Image_Processing.YOLO_ONNX_Format_Object_Detection(image_data, model, 640, 0.5)

        # ...
        if class_id_predicted_tmp != None:
            bounding_box_predicted = []
            for _, (bounding_box_predicted_tmp_i, confidence_predicted_tmp_i) in enumerate(zip(bounding_box_predicted_tmp, confidence_predicted_tmp)):
                bounding_box_predicted.append(list(bounding_box_predicted_tmp_i.values())); score_confidence.append(confidence_predicted_tmp_i)
        else:
            bounding_box_predicted = [[0] * 4]; score_confidence.append(0.0)
                
        # ...
        for _, bounding_box_desired_i in enumerate(bounding_box_desired):
            score_iou_tmp = []
            for _, bounding_box_predicted_i in enumerate(bounding_box_predicted):
                score_iou_tmp.append(torchvision.ops.boxes.box_iou(torch.tensor([bounding_box_desired_i], dtype=torch.float),
                                                                   torch.tensor([bounding_box_predicted_i], dtype=torch.float)).numpy()[0, 0])
            score_iou.append(Mathematics.Max(score_iou_tmp)[1])

    print(num_of_data)
    # ...
    mAP = np.sum(score_iou)/num_of_data
    print(mAP)

if __name__ == '__main__':
    sys.exit(main())