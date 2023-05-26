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
# Available objects.
#   ID{0} = 'T_Joint'
#   ID{1} = 'Metal_Blank'
CONST_OBJECT_NAME = ['T_Joint', 'Metal_Blank']
# The identification number of the dataset type.
CONST_DATASET_TYPE = 5
# Name of the dataset.
CONST_DATASET_NAME = f'Dataset_Type_{CONST_DATASET_TYPE}'
# The number of data to be tested for a single object.
CONST_NUM_OF_TEST_DATA = 30
# Initial iteration of the scanning process.
CONST_SCAN_ITERATION = 0
# The boundaries of the object area {A}.
#   The boundaries of each object were generated using this script:
#       ../Blender/gen_object_boundaries.py
#   imported into a Blender file:
#       ../Blender/Gen_Synthetic_Data.blend
CONST_BOUNDARIES_OBJECT_A = [[0.0056, 0.0112], [0.0045, 0.0129]]

def main():
    """
    Description:
        A program to display results based on object predictions from the test partition of a dataset. The prediction is performed using 
        the standard YOLO *.pt format.

        Note:
            The measurement mainly focuses on data such as Confidence, Intersection over Union (IoU) and Precision (P).
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Blender_Synthetic_Data')[0] + 'Blender_Synthetic_Data'

    # Load a pre-trained custom YOLO model.
    model = YOLO(f'{project_folder}/YOLO/Model/Type_{CONST_DATASET_TYPE}/yolov8n_custom.pt')

    # Set the parameters for the scientific style.
    plt.style.use('science')

    # Create a figure.
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(f'The name of the dataset: {CONST_DATASET_NAME}', fontsize = 30)

    # Tests the data up to the desired maximum number of iterations, which is given by the constant {CONST_NUM_OF_TEST_DATA}.
    s_iou = []
    for n_i in range(CONST_NUM_OF_TEST_DATA):
        # File path of the processed image.
        image_file_path = f'{project_folder}/Data/{CONST_DATASET_NAME}/images/test/Image_{(CONST_SCAN_ITERATION + (n_i + 1)):05}.png'

        # Loads images from the specified file.
        image_data = cv2.imread(image_file_path)

        # Loads labels (annotations) from the specified file.
        label_data = File_IO.Load(f'{project_folder}/Data/{CONST_DATASET_NAME}/labels/test/Image_{(CONST_SCAN_ITERATION + (n_i + 1)):05}', 'txt', ' ')

        # Convert bounding box data from YOLO format to PASCAL_VOC format.
        #   Note:
        #       Desired data to be used to validate results.
        b_box_des = []
        for _, label_data_i in enumerate(label_data):
            b_box_des_tmp = Lib.Utilities.General.Convert_Boundig_Box_Data('YOLO', 'PASCAL_VOC', 
                                                                           {'x_c': label_data_i[1], 
                                                                            'y_c': label_data_i[2], 
                                                                            'width': label_data_i[3], 
                                                                            'height': label_data_i[4]}, {'x': image_data.shape[1], 'y': image_data.shape[0]})
            b_box_des.append(list(b_box_des_tmp.values()))

        # Predict (test) the model on a test dataset.
        result = model.predict(source=image_file_path, imgsz=[480, 640], conf=0.5)

        # If the model has found an object in the current processed image, express the bounding box and the confidence of each object.
        #   Note:
        #       Predicted data to be used to validate results.
        s_conf = []
        if result[0].boxes.shape[0] >= 1:
            # Express the data from the prediction:
            #   ID name of the class, Bounding box in the YOLO format and Confidence.
            class_id_pred_tmp = result[0].boxes.cls.cpu().numpy(); b_box_pred_tmp = result[0].boxes.xyxy.cpu().numpy()
            b_box_pred_yolo_tmp = result[0].boxes.xywhn.cpu().numpy(); conf_pred_tmp = result[0].boxes.conf.cpu().numpy()

            # Check if the area of each predicted object is within the boundaries.
            b_box_pred = []
            for _, (class_id_pred_tmp_i, b_box_pred_tmp_i, conf_pred_tmp_i, b_box_pred_yolo_tmp_i) in enumerate(zip(class_id_pred_tmp, b_box_pred_tmp, 
                                                                                                                    conf_pred_tmp, b_box_pred_yolo_tmp)):
                # Calculates the area of the rectangle from the bounding box.
                A_tmp = b_box_pred_yolo_tmp_i[2] * b_box_pred_yolo_tmp_i[3]

                if CONST_BOUNDARIES_OBJECT_A[int(class_id_pred_tmp_i)][0] < A_tmp < CONST_BOUNDARIES_OBJECT_A[int(class_id_pred_tmp_i)][1]:
                    b_box_pred.append(b_box_pred_tmp_i); s_conf.append(conf_pred_tmp_i)
        else:
            # Otherwise, write null values for both the bounding box and the confidence.
            b_box_pred = [[0] * 4]; s_conf.append(0.0)

        # Find the Intersection over Union (IoU) score of the bounding boxes. 
        #   Note:
        #       If the number of predicted/desired objects is greater than one, calculate the IoU value of each pair 
        #       of objects (their combination) and find the maximum value.
        s_iou_tmp = []
        for _, b_box_des_i in enumerate(b_box_des):
            s_iou_i_tmp = []
            for _, b_box_pred_i in enumerate(b_box_pred):
                s_iou_i_tmp.append(torchvision.ops.boxes.box_iou(torch.tensor(np.array([b_box_des_i]), dtype=torch.float),
                                                                 torch.tensor(np.array([b_box_pred_i]), dtype=torch.float)).numpy()[0, 0])
            s_iou_tmp.append(Mathematics.Max(s_iou_i_tmp)[1])

        # Save the mean value (score) of the Intersection over Union (IoU). It will be used to calculate the mAP.
        s_iou.append(np.mean(s_iou_tmp))

        # Calculates the number of detected objects in the current episode. If the number of objects is greater than one, display the average 
        # of the data, otherwise display the original data.
        if len(b_box_des) == 1:
            ax.scatter(n_i + 1, s_conf, marker='o', color=[1.0,1.0,1.0,1.0], s=100.0, linewidth=3.0, 
                       edgecolors=[0.525,0.635,0.8,1.0], label='Confidence')
            ax.scatter(n_i + 1, s_iou_tmp, marker='o', color=[1.0,1.0,1.0,1.0], s=100.0, linewidth=3.0, 
                       edgecolors=[1.0,0.75,0.5,1.0], label='Intersection over Union (IoU)') 
        else:
            ax.scatter(n_i + 1, np.mean(s_conf), marker='o', color=[0.525,0.635,0.8,1.0], s=100.0, linewidth=3.0, 
                       edgecolors=[0.525,0.635,0.8,1.0], label='Average Confidence')
            ax.scatter(n_i + 1, np.mean(s_iou_tmp), marker='o', color=[1.0,0.75,0.5,1.0], s=100.0, linewidth=3.0, 
                       edgecolors=[1.0,0.75,0.5,1.0], label='Average Intersection over Union (AIoU)')  

    # Calculates the mean average precision (mAP).
    mAP = np.mean(np.array(s_iou, dtype=np.float32).flatten())
    print(f'Precision (P): {mAP}')

    # Display data of the mAP.
    ax.plot(np.arange(1, CONST_NUM_OF_TEST_DATA + 1, 1), [mAP] * CONST_NUM_OF_TEST_DATA, '--', color=[0.65,0.65,0.65,1.0], linewidth=2.0, ms = 10.0, 
            label='Precision (P)')

    # Set parameters of the graph (plot).
    #   Set the x ticks.
    ax.set_xticks(np.arange(1, CONST_NUM_OF_TEST_DATA + 1, 1))
    #   Label
    ax.set_xlabel(r'Image Idenfication Number', fontsize=15); ax.set_ylabel(r'Score', fontsize=15) 
    #   Set parameters of the visualization.
    ax.grid(which='major', linewidth = 0.25, linestyle = '--')
    # Get handles and labels for the legend.
    handles, labels = plt.gca().get_legend_handles_labels()
    # Remove duplicate labels.
    legend = dict(zip(labels, handles))
    # Show the labels (legends) of the graph.
    ax.legend(legend.values(), legend.keys(), fontsize=10.0)

    # Display the results as a graph (plot).
    plt.show()

if __name__ == '__main__':
    sys.exit(main())