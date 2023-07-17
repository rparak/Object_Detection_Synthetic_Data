# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../..')
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
# Number of datasets.
CONST_NUM_OF_DATASETS = 6
# Format of the trained model.
#   Standard YOLO *.pt format: 'PyTorch'
#   ONNX *.onnx format: 'ONNX'
CONST_MODEL_FORMAT = 'ONNX'

def main():
    """
    Description:
        A program to save validation results on a test dataset in the form of bar charts. Metrics such as Precision, Recall, Mean
        Average Precision (mAP), etc. were used to evaluate the performance of the proposed network.
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Object_Detection_Synthetic_Data')[0] + 'Object_Detection_Synthetic_Data'

    # Set the parameters for the scientific style.
    plt.style.use(['science'])

    metrics = []
    for i in range(CONST_NUM_OF_DATASETS):
        # The path to the trained model.
        if CONST_MODEL_FORMAT == 'PyTorch':
            file_path = f'{project_folder}/YOLO/Model/Type_{i}/yolov8n_custom.pt'
        elif CONST_MODEL_FORMAT == 'ONNX':
            file_path = f'{project_folder}/YOLO/Model/Type_{i}/yolov8n_dynamic_True_custom.onnx'

        # Load a pre-trained custom YOLO model in the desired format.
        model = YOLO(file_path)

        # Evaluate the performance of the model on the test dataset.
        results = model.val(data=f'{project_folder}/YOLO/Configuration/Type_{i}/config.yaml', batch=32, imgsz=640, conf=0.001, iou=0.6, rect=True, 
                            split='test')
        
        # Express the best metrics in the current dataset.
        metrics.append(list(results.results_dict.values())[0:-1])

    # Convert the list to an array.
    metrics = np.array(metrics, dtype=np.float32)

    # Create a figure.
    fig, ax = plt.subplots(1, 1)

    # Display metrics data in a bar chart.
    for i, color_i in enumerate(['#bfdbd1', '#72837d', '#abcae4', '#88a1b6', '#667988', '#a64d79']):
        ax.bar(np.arange(0, 4, 1) + i*0.05, metrics[i, :], color=color_i, alpha=1.0, width = 0.05, 
               label=f'Type {i}')

    # Set parameters of the graph (plot).
    #   Set the x, y ticks.
    plt.xticks(np.arange(0, 4, 1) + 0.125, ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95'])
    plt.yticks(np.arange(0, 1.1, 0.1))
    #   Set the y limits.
    ax.set(ylim=[0.8, 1.1])
    #   Label
    ax.set_xlabel(r'Metrics', fontsize=15); ax.set_ylabel(r'Score', fontsize=15) 
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
    fig.savefig(f'{project_folder}/images/Evaluation/Model/Validation_Comparison_{CONST_MODEL_FORMAT}.png', format='png', dpi=300)

if __name__ == '__main__':
    sys.exit(main())
