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
# Custom Library:
#   ../Lib/Utilities/Image_Processing
import Lib.Utilities.Image_Processing
#   ../Lib/Utilities/General
import Lib.Utilities.General

"""
Description:
    Initialization of constants.
"""
# Available objects.
#   ID{0} = 'T_Joint'
#   ID{1} = 'Metal_Blank'
CONST_OBJECT_NAME = ['T-Joint', 'Metal Blank']
# The color of the bounding box of the object.
CONST_OBJECT_BB_COLOR = [(255, 165, 0), (0, 165, 255)]
# The identification number of the dataset type.
CONST_DATASET_TYPE = 0
# Name of the dataset.
CONST_DATASET_NAME = f'Dataset_Type_{CONST_DATASET_TYPE}'
# The number of data to be tested for a single object.
CONST_NUM_OF_TEST_DATA = 25
# Initial iteration of the scanning process.
CONST_SCAN_ITERATION = 0

def main():
    """
    Description:
        A program to display results based on object predictions from the test partition of a dataset. The prediction is performed using 
        the *.onnx format.

        Note:
            The measurement mainly focuses on data such as Confidence, Intersection over Union (IoU) and Precision (P).
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Object_Detection_Synthetic_Data')[0] + 'Object_Detection_Synthetic_Data'

    # Load a pre-trained YOLO model in the *.onnx format.
    model = cv2.dnn.readNet(f'{project_folder}/YOLO/Model/Type_{CONST_DATASET_TYPE}/yolov8n_dynamic_False_custom.onnx')

    # Set the parameters for the scientific style.
    plt.style.use('science')

    # Create a figure.
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(f'The name of the dataset: {CONST_DATASET_NAME}', fontsize=30)

    # Tests the data up to the desired maximum number of iterations, which is given by the constant {CONST_NUM_OF_TEST_DATA}.
    a_conf_object_0 = []; a_conf_object_1 = []
    for n_i in range(CONST_NUM_OF_TEST_DATA):
        # File path of the processed image.
        image_file_path = f'{project_folder}/Data/{CONST_DATASET_NAME}/images/test/Image_{(CONST_SCAN_ITERATION + (n_i + 1)):05}.png'

        # Loads images from the specified file.
        image_data = cv2.imread(image_file_path)

        #   Express the data from the prediction:
        #       ID name of the class, Bounding box in the YOLO format and Confidence.
        (cls_id_pred_tmp, _, conf_pred_tmp) = Lib.Utilities.Image_Processing.YOLO_ONNX_Format_Object_Detection(image_data, model, 
                                                                                                               [640, 480], 0.5, 0.7)
        
        # If the model has found an object in the current processed image, express the bounding box and the confidence of each object.
        #   Note:
        #       Predicted data to be used to validate results.
        a_conf_object_0_tmp = []; a_conf_object_1_tmp = []
        if cls_id_pred_tmp != None:
            for _, (class_id_pred_tmp_i, conf_pred_tmp_i) in enumerate(zip(cls_id_pred_tmp, conf_pred_tmp)):
                if class_id_pred_tmp_i == 0:
                    ax.scatter(n_i + 1, conf_pred_tmp_i, marker='o', color=[1.0,1.0,1.0,1.0], s=100.0, linewidth=3.0, 
                               edgecolors=[1.0,0.75,0.5,1.0], label=f'{CONST_OBJECT_NAME[class_id_pred_tmp_i]}')
                    a_conf_object_0_tmp.append(conf_pred_tmp_i)
                elif class_id_pred_tmp_i == 1:
                    ax.scatter(n_i + 1, conf_pred_tmp_i, marker='o', color=[1.0,1.0,1.0,1.0], s=100.0, linewidth=3.0, 
                               edgecolors=[0.525,0.635,0.8,1.0], label=f'{CONST_OBJECT_NAME[class_id_pred_tmp_i]}')
                    a_conf_object_1_tmp.append(conf_pred_tmp_i)
        else:
            print('[INFO] The model did not find object in the input image.')
        
        # Calculates the average confidence of each object in the current episode.
        if a_conf_object_0_tmp:
            a_conf_object_0.append(np.mean(a_conf_object_0_tmp))
        if a_conf_object_1_tmp:
            a_conf_object_1.append(np.mean(a_conf_object_1_tmp))

    # Calculates the average confidence of each object.
    Average_Confindece_Object_0 = np.mean(np.array(a_conf_object_0, dtype=np.float32).flatten())
    print(f'[INFO] Average Confidence Score (T-Joint): {Average_Confindece_Object_0}')
    Average_Confindece_Object_1 = np.mean(np.array(a_conf_object_1, dtype=np.float32).flatten())
    print(f'[INFO] Average Confidence Score (Metal Blank): {Average_Confindece_Object_1}')
    Average_Confidence = (Average_Confindece_Object_0 + Average_Confindece_Object_1)/2.0
    print(f'[INFO] Average Confidence Score (ALL): {Average_Confidence}')

    # Display data of the average confidence of each object.
    ax.plot(np.arange(1, CONST_NUM_OF_TEST_DATA + 1, 1), [Average_Confindece_Object_0] * CONST_NUM_OF_TEST_DATA, '--', color=[1.0,0.75,0.5,1.0], linewidth=2.0, ms = 10.0, 
            label='Average: T-Joint')
    ax.plot(np.arange(1, CONST_NUM_OF_TEST_DATA + 1, 1), [Average_Confindece_Object_1] * CONST_NUM_OF_TEST_DATA, '--', color=[0.525,0.635,0.8,1.0], linewidth=2.0, ms = 10.0, 
            label='Average: Metal Blank')
    ax.plot(np.arange(1, CONST_NUM_OF_TEST_DATA + 1, 1), [Average_Confidence] * CONST_NUM_OF_TEST_DATA, '--', color=[0.65,0.65,0.65,1.0], linewidth=2.0, ms = 10.0, 
            label='Average: ALL')

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