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
# Ultralytics (Real-time object detection and image segmentation 
# model) [pip install ultralytics]
from ultralytics import YOLO
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
CONST_OBJECT_ID = 1
# Available objects.
#   ID{0} = 'T_Joint'
#   ID{1} = 'Metal_Blank'
CONST_OBJECT_NAME = ['T_Joint', 'Metal_Blank']
# The identification number of the dataset type.
CONST_DATASET_TYPE = 5
# Name of the dataset.
CONST_DATASET_NAME = f'Dataset_Type_{CONST_DATASET_TYPE}'
# Number of data to be tested.
CONST_NUM_OF_TEST_DATA = 15
# Initial iteration of the scanning process.
CONST_SCAN_ITERATION = 30
# The boundaries of the object area {A}.
CONST_BOUNDARIES_OBJECT_A = [[0.0056, 0.0112], [0.0045, 0.0129]]

def main():
    """
    Description:
        ...
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Blender_Synthetic_Data')[0] + 'Blender_Synthetic_Data'

    # Load a pre-trained custom YOLO model.
    model = YOLO(f'{project_folder}/YOLO/Model/Type_{CONST_DATASET_TYPE}/yolov8n_custom.pt')

    score_confidence = []; score_iou = []
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

        # Calculate the number of data in the current row
        num_of_data = len(bounding_box_desired)

        # Predict (test) the model on a test dataset.
        result = model.predict(source=image_file_path, imgsz=[480, 640], conf=0.5)

        # ...
        score_confidence_tmp = []
        if result[0].boxes.shape[0] >= 1:
            bounding_box_predicted_tmp = result[0].boxes.xyxy.cpu().numpy()
            bounding_box_predicted_yolo_tmp = result[0].boxes.xywhn.cpu().numpy()
            confidence_predicted_tmp   = result[0].boxes.conf.cpu().numpy()
            bounding_box_predicted = []
            for _, (bounding_box_predicted_tmp_i, bounding_box_predicted_yolo_tmp_i, confidence_predicted_tmp_i) in enumerate(zip(bounding_box_predicted_tmp, bounding_box_predicted_yolo_tmp, 
                                                                                                                                  confidence_predicted_tmp)):
                if CONST_BOUNDARIES_OBJECT_A[CONST_OBJECT_ID][0] < \
                   bounding_box_predicted_yolo_tmp_i[2] * bounding_box_predicted_yolo_tmp_i[3] < \
                   CONST_BOUNDARIES_OBJECT_A[CONST_OBJECT_ID][1]:
                    bounding_box_predicted.append(bounding_box_predicted_tmp_i); score_confidence_tmp.append(confidence_predicted_tmp_i)
        else:
            bounding_box_predicted = [[0] * 4]; score_confidence_tmp.append(0.0)

        #   ...
        score_confidence.append(np.sum(score_confidence_tmp)/num_of_data)

        # ...
        score_iou_tmp = []
        for _, bounding_box_desired_i in enumerate(bounding_box_desired):
            score_iou_i_tmp = []
            for _, bounding_box_predicted_i in enumerate(bounding_box_predicted):
                score_iou_i_tmp.append(torchvision.ops.boxes.box_iou(torch.tensor(np.array([bounding_box_desired_i]), dtype=torch.float),
                                                                     torch.tensor(np.array([bounding_box_predicted_i]), dtype=torch.float)).numpy()[0, 0])
            score_iou_tmp.append(Mathematics.Max(score_iou_i_tmp)[1])
        #   ..
        score_iou.append(np.mean(score_iou_tmp))

    # ...
    mAP = np.mean(np.array(score_iou, dtype=np.float32).flatten())
    print(f'Mean Average Precision (mAP): {mAP}')

    # ...
    image_number = []; average_iou = []; average_confidence = []
    for i, (score_iou_i, score_confidence_i) in enumerate(zip(score_iou, score_confidence)):
        # ...
        image_number.append(i + 1)
        # ...
        average_iou.append(score_iou_i); average_confidence.append(score_confidence_i)

    # Set the parameters for the scientific style.
    plt.style.use('science')

    # Create a figure with 5 subplots.
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(f'The name of the dataset: {CONST_DATASET_NAME}\nClass: ID = {CONST_OBJECT_ID}, Name = {CONST_OBJECT_NAME[CONST_OBJECT_ID]}', fontsize = 30)

    # Display data ....
    ax.plot(image_number[0:10], average_confidence[0:10], 'o', color=[0.525,0.635,0.8,1.0], linewidth=2.0, ms = 10.0, mfc = [0.525,0.635,1.0], markeredgewidth = 5,
            label='Confidence')
    ax.plot(image_number[0:10], average_iou[0:10], 'o', color=[1.0,0.75,0.5,1.0], linewidth=2.0, ms = 10.0, mfc = [1.0,0.75,0.5,1.0], markeredgewidth = 5,
            label='Intersection over Union (IoU)')
    ax.plot(image_number[9::], average_confidence[9::], 'o', color=[0.525,0.635,0.8,1.0], linewidth=2.0, ms = 10.0, mfc = [1.0,1.0,1.0,1.0], markeredgewidth = 5,
            label='Average Confidence')
    ax.plot(image_number[9::], average_iou[9::], 'o', color=[1.0,0.75,0.5,1.0], linewidth=2.0, ms = 10.0, mfc = [1.0,1.0,1.0,1.0], markeredgewidth = 5, 
            label='Average Intersection over Union (IoU)')
    ax.plot(image_number, [mAP] * len(image_number), '--', color=[0.65,0.65,0.65,1.0], linewidth=2.0, ms = 10.0, label='Mean Average Precision (mAP)')
    #   Set the x ticks.
    ax.set_xticks(np.arange(1, len(image_number) + 1, 1))
    #   Label
    ax.set_xlabel(r'Image Idenfication Number (ID)'); ax.set_ylabel(r'Score') 
    #   Set parameters of the visualization.
    ax.grid(which='major', linewidth = 0.25, linestyle = '--')
    ax.legend(fontsize=10.0)

    # Display the results as a graph (plot).
    plt.show()

if __name__ == '__main__':
    sys.exit(main())