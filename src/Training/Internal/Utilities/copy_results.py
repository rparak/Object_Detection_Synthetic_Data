# OS (Operating system interfaces)
import os
# Shutil (High-level file operations)
import shutil

"""
Description:
    Initialization of constants.
"""
# Number of datasets.
CONST_NUM_OF_DATASETS = 6
# An indication of whether the backbone layers of the model should be frozen.
CONST_FREEZE_BACKBONE = True

def main():
    """
    Description:
        Copy the training results (*.csv) from the train folder to the desired folder to be used in the future.
    """
    
    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Object_Detection_Synthetic_Data')[0] + 'Object_Detection_Synthetic_Data'

    for i in range(CONST_NUM_OF_DATASETS):
        # The specified path of the file.
        file_path_old = f'{project_folder}/YOLO/Results/Type_{i}/train_fb_{CONST_FREEZE_BACKBONE}/'
        file_path_new = f'{project_folder}/YOLO/Results/Type_{i}/'
        
        if os.path.isfile(file_path_new + 'results.csv'):
            print('[INFO] The file already exists..')
        else:
            # Copy the file.
            shutil.copy(file_path_old + 'results.csv', file_path_new)
            print(f'[INFO] The file results.csv was successfully copied.')
            print(f'[INFO]  - In: {file_path_old}')
            print(f'[INFO]  - Out: {file_path_new}')

if __name__ == '__main__':
    main()