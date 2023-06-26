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
        A program to remove unnecessary directories (results) created from scripts:
            ./train.py   - train_fb_True
            ./valid.py   - valid_fb_True
            ./predict.py - predict_fb_True
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Object_Detection_Synthetic_Data')[0] + 'Object_Detection_Synthetic_Data'

    for i in range(CONST_NUM_OF_DATASETS):
        for _, partition in enumerate(['train', 'valid', 'predict']):
            file_path = f'{project_folder}/YOLO/Results/Type_{i}/{partition}_fb_{CONST_FREEZE_BACKBONE}'
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
                print(f'[INFO] The directory has been successfully removed.')
                print(f'[INFO]  - Path: {file_path}')

if __name__ == '__main__':
    main()