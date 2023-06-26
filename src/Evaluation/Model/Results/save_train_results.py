# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../..')
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
# The identification number of the dataset type.
CONST_DATASET_TYPE = 5
# Name of the dataset.
CONST_DATASET_NAME = f'Type_{CONST_DATASET_TYPE}'

def main():
    """
    Description:
        A program to save result data from training a dataset. Metrics such as Generalized Intersection over Union (GIoU), Mean
        Average Precision (mAP), Precision, etc. were used to evaluate the performance of the proposed network.
    """
        
    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Object_Detection_Synthetic_Data')[0] + 'Object_Detection_Synthetic_Data'

    # The specified path of the file.
    file_path = f'{project_folder}/YOLO/Results/Type_{CONST_DATASET_TYPE}'  

    # Read data from the file {*.csv}.
    data = pd.read_csv(f'{file_path}/results.csv')

    # Assign data to variables.
    #   The total number of iterations of the training data.
    epoch = data[data.columns[0]]

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

    # Display the results as the values shown in the console.
    print('[INFO] Evaluation Criteria: YOLOv8')
    print(f'[INFO] The name of the dataset: {CONST_DATASET_NAME}')
    print(f'[INFO] The best results were found in the {idx} iteration.')
    print('[INFO]  Generalized Intersection over Union (GIoU):')
    print(f'[INFO]  [train = {data[data.columns[1]][idx]}, valid = {data[data.columns[8]][idx]}]')
    print('[INFO]  Objectness:')
    print(f'[INFO]  [train = {data[data.columns[2]][idx]}, valid = {data[data.columns[9]][idx]}]')
    print('[INFO]  Classification:')
    print(f'[INFO]  [train = {data[data.columns[3]][idx]}, valid = {data[data.columns[10]][idx]}]')
    print('[INFO]  Pr + Rec:')
    print(f'[INFO]  [precision = {data[data.columns[4]][idx]}, recall = {data[data.columns[5]][idx]}]')
    print('[INFO]  Mean Average Precision (mAP):')
    print(f'[INFO]  [mAP@0.5 = {data[data.columns[6]][idx]}, mAP@0.5:0.95 = {data[data.columns[7]][idx]}]')
    
    # Set the parameters for the scientific style.
    plt.style.use(['science'])

    # Create a figure with 5 subplots.
    fig, ax = plt.subplots(1, 5)
    fig.suptitle(f'The name of the Dataset: {CONST_DATASET_NAME}', fontsize=25)

    # Generalized Intersection over Union (GIoU)
    ax[0].plot(epoch, data[data.columns[1]], 'o-', color=[0.525,0.635,0.8,1.0], linewidth=2.0, ms = 3.0, label='train')
    ax[0].plot(epoch, data[data.columns[8]], 'o-', color=[1.0,0.75,0.5,1.0], linewidth=2.0, ms = 3.0, label='valid')
    #   Set parameters of the visualization.
    ax[0].set_title('GIoU', fontsize = 15)
    ax[0].grid(linewidth = 0.25, linestyle = '--')
    ax[0].legend(fontsize=10.0)

    # Objectness.
    ax[1].plot(epoch, data[data.columns[2]], 'o-', color=[0.525,0.635,0.8,1.0], linewidth=2.0, ms = 3.0, label='train')
    ax[1].plot(epoch, data[data.columns[9]], 'o-', color=[1.0,0.75,0.5,1.0], linewidth=2.0, ms = 3.0, label='valid')
    #   Set parameters of the visualization.
    ax[1].set_title('Objectness', fontsize = 15)
    ax[1].grid(linewidth = 0.25, linestyle = '--')
    ax[1].legend(fontsize=10.0)

    # Classification.
    ax[2].plot(epoch, data[data.columns[3]], 'o-', color=[0.525,0.635,0.8,1.0], linewidth=2.0, ms = 3.0, label='train')
    ax[2].plot(epoch, data[data.columns[10]], 'o-', color=[1.0,0.75,0.5,1.0], linewidth=2.0, ms = 3.0, label='valid')
    #   Set parameters of the visualization.
    ax[2].set_title('Classification', fontsize = 15)
    ax[2].grid(linewidth = 0.25, linestyle = '--')
    ax[2].legend(fontsize=10.0)

    # Precision + Recall (Pr + Rec).
    ax[3].plot(epoch, data[data.columns[4]], 'o-', color=[0.525,0.635,0.8,1.0], linewidth=2.0, ms = 3.0, label='precision')
    ax[3].plot(epoch, data[data.columns[5]], 'o-', color=[1.0,0.75,0.5,1.0], linewidth=2.0, ms = 3.0, label='recall')
    #   Set parameters of the visualization.
    ax[3].set_title('Pr. + Rec.', fontsize = 15)
    ax[3].grid(linewidth = 0.25, linestyle = '--')
    ax[3].legend(fontsize=10.0)

    # Mean Average Precision (mAP) - (mAP@0.5, mAP@0.5:0.95).
    ax[4].plot(epoch, data[data.columns[6]], 'o-', color=[0.525,0.635,0.8,1.0], linewidth=2.0, ms = 3.0, label='mAP@0.5')
    ax[4].plot(epoch, data[data.columns[7]], 'o-', color=[1.0,0.75,0.5,1.0], linewidth=2.0, ms = 3.0, label='mAP@0.5:0.95')
    #   Set parameters of the visualization.
    ax[4].set_title('mAP', fontsize = 15)
    ax[4].grid(linewidth = 0.25, linestyle = '--')
    ax[4].legend(fontsize=10.0)

    # Set the full scree mode.
    plt.get_current_fig_manager().full_screen_toggle()

    # Save the results.
    fig.savefig(f'{project_folder}/images/Evaluation/Model/{CONST_DATASET_NAME}/Training_Results.png', format='png', dpi=300)

if __name__ == '__main__':
    sys.exit(main())