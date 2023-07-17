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
# Shutil (High-level file operations)
import shutil
# SciencePlots (Matplotlib styles for scientific plotting) [pip3 install SciencePlots]
import scienceplots
# Ultralytics (Real-time object detection and image segmentation 
# model) [pip install ultralytics]
from ultralytics import YOLO
# Custom Library:
#   ../Lib/Utilities/File_IO
import Lib.Utilities.File_IO as File_IO

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

def main():
    """
    Description:
        A program to save the results based on the validation of the YOLOv8 model. In this case, the model is evaluated 
        on a test dataset to measure metrics such as Precision, Recall, mAP@0.5, mAP@0.5:0.95.

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

    # The name of the temporary folder with the validation results.
    tmp_folder_name = f'{project_folder}/src/Evaluation/Model/Results/tmp_folder'

    # Load a pre-trained custom YOLO model in the desired format.
    model = YOLO(file_path)

    # Evaluate the performance of the model on the test dataset.
    results = model.val(data=f'{project_folder}/YOLO/Configuration/Type_{CONST_DATASET_TYPE}/config.yaml', batch=32, imgsz=640, conf=0.001, iou=0.6, rect=True, 
                        save_txt=True, save_conf=True, save_json=False, split='test', name=tmp_folder_name)
    
    # Express the best metrics in the current dataset.
    metrics = list(results.results_dict.values())[0:-1]
    
    # Set the parameters for the scientific style.
    plt.style.use('science')

    # Create a figure.
    fig, ax = plt.subplots(1, 1)

    for i in range(CONST_NUM_OF_TEST_DATA):
        # Load a label (annotation) from a file.
        label_data = File_IO.Load(f'{tmp_folder_name}/labels/Image_{(i + 1):05}', 'txt', ' ')
        for _, label_data_i in enumerate(label_data):
            cls_id = int(label_data_i[0]); conf = label_data_i[-1]
            if cls_id == 0:
                ax.scatter(i + 1, conf, marker='o', color=[1.0,1.0,1.0,1.0], s=100.0, linewidth=3.0, 
                           edgecolors=[1.0,0.75,0.5,1.0], label=f'{CONST_OBJECT_NAME[cls_id]}')
            elif cls_id == 1:
                ax.scatter(i + 1, conf, marker='o', color=[1.0,1.0,1.0,1.0], s=100.0, linewidth=3.0, 
                           edgecolors=[0.525,0.635,0.8,1.0], label=f'{CONST_OBJECT_NAME[cls_id]}')

    # Remove the unnecessary (temporary) folder.
    if os.path.isdir(tmp_folder_name):
        shutil.rmtree(tmp_folder_name)
        
    # Display data of the monitored metrics.
    print('[INFO] Evaluation Criteria: YOLOv8')
    print(f'[INFO] The name of the dataset: {CONST_DATASET_NAME}')
    for _, (metrics_data, metrics_name, color_i) in enumerate(zip(metrics, ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95'], 
                                                              ['#bfdbd1', '#98afa7', '#72837d', '#39413e'])):
        ax.plot(np.arange(1, CONST_NUM_OF_TEST_DATA + 1, 1), [metrics_data] * CONST_NUM_OF_TEST_DATA, '--', color=color_i, linewidth=2.0, ms = 10.0, 
                label=f'{metrics_name}')
        # Display the results as the values shown in the console.
        print(f'[INFO] {metrics_name} = {metrics_data}')

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
    fig.savefig(f'{project_folder}/images/Evaluation/Model/Type_{CONST_DATASET_TYPE}/{CONST_MODEL_FORMAT}/Validation_0001_Conf_06_IOU.png', format='png', dpi=300)

if __name__ == '__main__':
    sys.exit(main())
