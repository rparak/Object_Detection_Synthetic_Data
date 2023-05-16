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
# Custom Library:
#   ../Lib/Transformation/Utilities/Mathematics
import Lib.Transformation.Utilities.Mathematics as Mathematics

"""
Description:
    Initialization of constants.
"""
# The identification number of the dataset type.
CONST_DATASET_TYPE = 0
# Name of the dataset.
CONST_DATASET_NAME = f'Dataset_Type_{CONST_DATASET_TYPE}'

def main():
    """
    Description:
        A simple script to display result data from training a dataset. Metrics such as Generalized Intersection over Union (GIoU), Mean
        Average Precision (mAP), Precision, etc. were used to evaluate the performance of the proposed network.
    """
        
    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Blender_Synthetic_Data')[0] + 'Blender_Synthetic_Data'

    # The specified path of the file.
    file_path = f'{project_folder}/YOLO/Results/Type_{CONST_DATASET_TYPE}'  

    # Read data from the file {*.csv}.
    data = pd.read_csv(f'{file_path}/results.csv')

    # Assign data to variables.
    #   The total number of iterations of the training data.
    epoch = data[data.columns[0]]

    # Set the parameters for the scientific style.
    plt.style.use('science')

    # Create a figure with 5 subplots.
    fig, ax = plt.subplots(1, 5)
    fig.suptitle(f'The name of the dataset: {CONST_DATASET_NAME}', fontsize = 20)

    # GloU.
    ax[0].plot(epoch, data[data.columns[1]], 'o-', color=[0.525,0.635,0.8,1.0], linewidth=2.0, ms = 3.0, label='train')
    ax[0].plot(epoch, data[data.columns[8]], 'o-', color=[1.0,0.75,0.5,1.0], linewidth=2.0, ms = 3.0, label='valid')
    #   Set parameters of the visualization.
    ax[0].set_title('GIoU')
    ax[0].grid(linewidth = 0.75, linestyle = '--')
    ax[0].legend(fontsize=10.0)

    # Objectness.
    ax[1].plot(epoch, data[data.columns[2]], 'o-', color=[0.525,0.635,0.8,1.0], linewidth=2.0, ms = 3.0, label='train')
    ax[1].plot(epoch, data[data.columns[9]], 'o-', color=[1.0,0.75,0.5,1.0], linewidth=2.0, ms = 3.0, label='valid')
    #   Set parameters of the visualization.
    ax[1].set_title('Objectness')
    ax[1].grid(linewidth = 0.75, linestyle = '--')
    ax[1].legend(fontsize=10.0)

    # Classification.
    ax[2].plot(epoch, data[data.columns[3]], 'o-', color=[0.525,0.635,0.8,1.0], linewidth=2.0, ms = 3.0, label='train')
    ax[2].plot(epoch, data[data.columns[10]], 'o-', color=[1.0,0.75,0.5,1.0], linewidth=2.0, ms = 3.0, label='valid')
    #   Set parameters of the visualization.
    ax[2].set_title('Classification')
    ax[2].grid(linewidth = 0.75, linestyle = '--')
    ax[2].legend(fontsize=10.0)

    # Precision + Recall (Pr + Rec).
    ax[3].plot(epoch, data[data.columns[4]], 'o-', color=[0.525,0.635,0.8,1.0], linewidth=2.0, ms = 3.0, label='precision')
    ax[3].plot(epoch, data[data.columns[5]], 'o-', color=[1.0,0.75,0.5,1.0], linewidth=2.0, ms = 3.0, label='recall')
    #   Set parameters of the visualization.
    ax[3].set_title('Pr + Rec')
    ax[3].grid(linewidth = 0.75, linestyle = '--')
    ax[3].legend(fontsize=10.0)

    # mAP (mAP@0.5, mAP@0.5:0.95)
    ax[4].plot(epoch, data[data.columns[6]], 'o-', color=[0.525,0.635,0.8,1.0], linewidth=2.0, ms = 3.0, label='mAP@0.5')
    ax[4].plot(epoch, data[data.columns[7]], 'o-', color=[1.0,0.75,0.5,1.0], linewidth=2.0, ms = 3.0, label='mAP@0.5:0.95')
    #   Set parameters of the visualization.
    ax[4].set_title('mAP')
    ax[4].grid(linewidth = 0.75, linestyle = '--')
    ax[4].legend(fontsize=10.0)

    # Display the results as a graph (plot).
    plt.show()

    # Display the results as the values shown in the console.
    print('[INFO] Evaluation Criteria: YOLOv8')
    print(f'[INFO] The name of the dataset: {CONST_DATASET_NAME}')
    print('[INFO]  Generalized Intersection over Union (GIoU)')
    print(f'[INFO]  - train = {Mathematics.Min(data[data.columns[1]])[1]}')
    print(f'[INFO]  - valid = {Mathematics.Min(data[data.columns[8]])[1]}')
    print('[INFO]  Objectness')
    print(f'[INFO]  - train = {Mathematics.Min(data[data.columns[2]])[1]}')
    print(f'[INFO]  - valid = {Mathematics.Min(data[data.columns[9]])[1]}')
    print('[INFO]  Classification')
    print(f'[INFO]  - train = {Mathematics.Min(data[data.columns[3]])[1]}')
    print(f'[INFO]  - valid = {Mathematics.Min(data[data.columns[10]])[1]}')
    print('[INFO]  Pr + Rec')
    print(f'[INFO]  - precision = {Mathematics.Max(data[data.columns[4]])[1]}')
    print(f'[INFO]  - recall = {Mathematics.Max(data[data.columns[5]])[1]}')
    print('[INFO]  Mean Average Precision (mAP)')
    print(f'[INFO]  - mAP@0.5 = {Mathematics.Max(data[data.columns[6]])[1]}')
    print(f'[INFO]  - mAP@0.5:0.95 = {Mathematics.Max(data[data.columns[7]])[1]}')

if __name__ == '__main__':
    sys.exit(main())