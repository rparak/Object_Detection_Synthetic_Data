# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../../../')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Matplotlib (Visualization) [pip3 install matplotlib]
import matplotlib.pyplot as plt
# OS (Operating system interfaces)
import os
# SciencePlots (Matplotlib styles for scientific plotting) [pip3 install SciencePlots]
import scienceplots
# Ultralytics (Real-time object detection and image segmentation 
# model) [pip install ultralytics]
from ultralytics import YOLO

"""
Description:
    Initialization of constants.
"""
# The identification number of the dataset type.
CONST_DATASET_TYPE = 0
# Name of the dataset.
CONST_DATASET_NAME = f'Dataset_Type_{CONST_DATASET_TYPE}'
# Available objects.
#   ID{0} = 'T_Joint'
#   ID{1} = 'Metal_Blank'
CONST_OBJECT_NAME = ['T-Joint', 'Metal Blank']
# Format of the trained model.
#   Standard YOLO *.pt format: 'PyTorch'
#   ONNX *.onnx format: 'ONNX'
CONST_MODEL_FORMAT = 'PyTorch'
# The number of data to be tested for a single object.
CONST_NUM_OF_TEST_DATA = 25
# Initial iteration of the scanning process.
CONST_SCAN_ITERATION = 0

def main():
    """
    Description:
        A program to save the results based on object predictions from the test partition of a dataset.

        Note:
            The prediction is performed using both *.onnx and *.pt formats.
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Object_Detection_Synthetic_Data')[0] + 'Object_Detection_Synthetic_Data'

    # The path to the trained model.
    if CONST_MODEL_FORMAT == 'PyTorch':
        file_path = f'{project_folder}/YOLO/Model/Type_{CONST_DATASET_TYPE}/yolov8n_custom.pt'
    elif CONST_MODEL_FORMAT == 'ONNX':
        file_path = f'{project_folder}/YOLO/Model/Type_{CONST_DATASET_TYPE}/yolov8n_dynamic_True_custom.onnx'

    # Set the parameters for the scientific style.
    plt.style.use('science')

    # Create a figure.
    fig, ax = plt.subplots(1, 1)

    # Load a pre-trained custom YOLO model in the desired format.
    model = YOLO(file_path)

    # Tests the data up to the desired maximum number of iterations, which is given by the constant {CONST_NUM_OF_TEST_DATA}.
    for n_i in range(CONST_NUM_OF_TEST_DATA):
        # File path of the processed image.
        image_file_path = f'{project_folder}/Data/{CONST_DATASET_NAME}/images/test/Image_{(CONST_SCAN_ITERATION + (n_i + 1)):05}.png'

        # Evaluate the performance of the model on the test dataset.
        results = model.predict(source=image_file_path, imgsz=[480, 640], conf=0.5, iou=0.7)
        
        # If the model has found an object in the current processed image, express the results (class, confidence).
        if results[0].boxes.shape[0] >= 1:
            # Express the data from the prediction:
            cls_id = results[0].boxes.cls.cpu().numpy(); conf = results[0].boxes.conf.cpu().numpy()
            
            for _, (cls_id_i, conf_i) in enumerate(zip(cls_id, conf)):
                if cls_id_i == 0:
                    ax.scatter(n_i + 1, conf_i, marker='o', color=[1.0,1.0,1.0,1.0], s=100.0, linewidth=3.0, 
                               edgecolors=[1.0,0.75,0.5,1.0], label=f'{CONST_OBJECT_NAME[int(cls_id_i)]}')
                elif cls_id_i == 1:
                    ax.scatter(n_i + 1, conf_i, marker='o', color=[1.0,1.0,1.0,1.0], s=100.0, linewidth=3.0, 
                               edgecolors=[0.525,0.635,0.8,1.0], label=f'{CONST_OBJECT_NAME[int(cls_id_i)]}')
                    
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

    # Set the full scree mode.
    plt.get_current_fig_manager().full_screen_toggle()

    # Save the results.
    fig.savefig(f'{project_folder}/images/Evaluation/Model/Type_{CONST_DATASET_TYPE}/{CONST_MODEL_FORMAT}/Prediction_05_Conf_07_IOU.png', format='png', dpi=300)

if __name__ == '__main__':
    sys.exit(main())
