# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../..')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Pandas (Data analysis and manipulation) [pip3 install pandas]
import pandas as pd
# Matplotlib (Visualization) [pip3 install matplotlib]
import matplotlib.pyplot as plt
# OS (Operating system interfaces)
import os
# SciencePlots (Matplotlib styles for scientific plotting) [pip3 install SciencePlots]
import scienceplots
 
"""
Description:
    Initialization of constants.
"""
# Number of datasets.
CONST_NUM_OF_DATASETS = 6

def main():
    """
    Description:
        A program to save result data from training a dataset in the form of bar charts. Metrics such as Generalized Intersection over Union (GIoU), Mean
        Average Precision (mAP), Precision, etc. were used to evaluate the performance of the proposed network.
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Object_Detection_Synthetic_Data')[0] + 'Object_Detection_Synthetic_Data'

    # Set the parameters for the scientific style.
    plt.style.use(['science'])

    metrics = []
    for i in range(CONST_NUM_OF_DATASETS):
        # Read data from the file {*.csv}.
        data = pd.read_csv(f'{project_folder}/YOLO/Results/Type_{i}/results.csv').copy()

        # Get the YOLOv* model with the best loss selected by the trainer (best.pt).
        #   The fitness function is defined as a weighted combination of metrics: 
        #       - mAP@0.5: 10% 
        #       - mAP@0.5:0.95: 90%
        #   Precision {P} and Recall {R} are missing. 
        #
        #   The fitness function can be found at:
        #       https://github.com/ultralytics/ultralytics/../utils/metrics.py
        best_fitness = data[data.columns[6]][0]*0.1 + data[data.columns[7]][0]*0.9; idx = 0
        for i, (mAP_i, mAP_95_i) in enumerate(zip(data[data.columns[6]][1:], data[data.columns[7]][1:])):
            best_fitness_tmp = mAP_i*0.1 + mAP_95_i*0.9
            if best_fitness_tmp > best_fitness:
                best_fitness = best_fitness_tmp
                idx = i + 1
        
        # Express the best metrics in the current dataset.
        #   Generalized Intersection over Union (GIoU): (1, 8)
        #   Objectness and Classification: (2, 9), (3, 10)
        #   Precision + Recall (Pr + Rec): (4, 5)
        #   Mean Average Precision (mAP) - (mAP@0.5, mAP@0.5:0.95): (6, 7)
        metrics.append([[data[data.columns[1]][idx], data[data.columns[10]][idx]],
                        [data[data.columns[2]][idx], data[data.columns[4]][idx]],
                        [data[data.columns[3]][idx], data[data.columns[5]][idx]],
                        [data[data.columns[8]][idx], data[data.columns[6]][idx]],
                        [data[data.columns[9]][idx], data[data.columns[7]][idx]]])

    # Convert the list to an array.
    metrics = np.array(metrics, dtype=np.float32)

    # Create a figure.
    fig, ax = plt.subplots(1, 1)

    # Display metrics data in a bar chart.
    for i, color_i in enumerate(['#bfdbd1', '#72837d', '#abcae4', '#88a1b6', '#667988', '#a64d79']):
        ax.bar(np.arange(0, 10, 1) + i*0.05, np.array([metrics[i, :, 0], metrics[i, :, 1]]).flatten(), color=color_i, alpha=1.0, width = 0.05,
               label=f'Type {i}')

    # Set parameters of the graph (plot).
    #   Set the x,y ticks.
    plt.xticks(np.arange(0, 10, 1) + 0.125, ['GIoU: train', 'Objectness: train', 'Classification: train', 
                                             'GIoU: valid', 'Objectness: valid', 'Classification: valid',
                                             'Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95'])
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    #   Set the y limits.
    ax.set(ylim=[0.0, 1.1])
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
    fig.savefig(f'{project_folder}/images/Evaluation/Model/Training_Comparison.png', format='png', dpi=300)

if __name__ == '__main__':
    sys.exit(main())