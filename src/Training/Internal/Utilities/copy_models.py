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
# Select the desired size of YOLOv* to build the model.
#   Note:
#     Detection Model.
#   Nano: 'yolov8n', Small: 'yolov8s', Medium: 'yolov8m', Large: 'yolov8l', XLarge: 'yolov8x'}
CONST_YOLO_SIZE = 'yolov8n'
# An indication of whether the backbone layers of the model should be frozen.
CONST_FREEZE_BACKBONE = True

def main():
    """
    Description:
        Copy the pre-trained model (*.pt) from the train folder to the desired folder to be used in the future.
    """
    
    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Object_Detection_Synthetic_Data')[0] + 'Object_Detection_Synthetic_Data'

    for i in range(CONST_NUM_OF_DATASETS):
        # The specified path of the file.
        file_path_old = f'{project_folder}/YOLO/Results/Type_{i}/train_fb_{CONST_FREEZE_BACKBONE}/weights/'
        file_path_new = f'{project_folder}/YOLO/Model/Type_{i}/'
        
        # Name of the pre-trained model.
        model_name_old = 'best.pt'
        model_name_new = f'{CONST_YOLO_SIZE}_custom.pt'

        if os.path.isfile(file_path_new + model_name_old):
            print('[INFO] The file already exists..')
        else:
            # Copy the file.
            shutil.copy(file_path_old + model_name_old, file_path_new)
            print(f'[INFO] The file ({model_name_old}) was successfully copied.')
            print(f'[INFO]  - In: {file_path_old}')
            print(f'[INFO]  - Out: {file_path_new}')

            # Rename the file.
            os.rename(file_path_new + model_name_old, file_path_new + model_name_new)
            print(f'[INFO] The file {model_name_old} was successfully renamed to {model_name_new}.')

if __name__ == '__main__':
    main()