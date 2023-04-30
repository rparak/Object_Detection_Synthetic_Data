# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('..')
# OS (Operating system interfaces)
import os
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# OpenCV (Computer Vision) [pip3 install opencv-python]
import cv2
# Albumentations (Library for image augmentation) [pip3 install albumentations]
import albumentations as A
# Custom Library:
#   ../Lib/Utilities/File_IO
import Lib.Utilities.File_IO as File_IO

"""
Description:
    Initialization of constants.
"""
# The ID of the object to be processed.
CONST_OBJECT_ID = 0
# Available objects.
#   ID{0} = 'T_Joint'
#   ID{1} = 'Metal_Blank'
CONST_OBJECT_NAME = ['T_Joint', 'Metal_Blank']
# The identification number of the dataset type.
CONST_DATASET_TYPE = 2
# Name of the dataset.
CONST_DATASET_NAME = F'Dataset_Type_{CONST_DATASET_TYPE}_Obj_ID_{CONST_OBJECT_ID}'
# Number of augmented data to be generated.
CONST_NUM_OF_GEN_DATA = 270
# Explanation of the dictionary of dataset partitions.
#   'Percentage': Partition the dataset into training, validation, and test sets in percentages.
#                 Note:
#                   The sum of the values in the partitions must equal 100.
#   'Init_Index': The initial index (iteration) of the augmented data in each partition.
#                 Note:
#                   The value of the initial index is equal to the last image/label value 
#                   in the dataset + 1.
#   'Num_of_Data': Number of data in each partitions.
CONST_DATASET_PARTITION = {'Percentage': [80, 20, 0], 'Init_Index': [25, 247, 0], 'Num_of_Data': [24, 6, 0]}

def main():
    """
    Descriptioon:
        ...
    """

    # Locate the path to the project folder
    project_folder = os.getcwd().split('Blender_Synthetic_Data')[0] + 'Blender_Synthetic_Data'

    try:
        assert np.sum(CONST_DATASET_PARTITION['Percentage']) == 100

        # ...
        i = 0; id_partition = 0; percentage_stored_data = 0
        while CONST_NUM_OF_GEN_DATA > i:
            # Calculate the current percentage of stored data.
            if percentage_stored_data == (CONST_NUM_OF_GEN_DATA * (CONST_DATASET_PARTITION['Percentage'])[id_partition]/100):
                id_partition += 1; percentage_stored_data = 0
                if CONST_DATASET_PARTITION['Percentage'][id_partition] == 0:
                    id_partition += 1

            # Describe the current iteration for loading/saving data (image/label).
            iter_load_data = (percentage_stored_data % CONST_DATASET_PARTITION['Num_of_Data'][id_partition]) \
                           + (CONST_DATASET_PARTITION['Init_Index'][id_partition] - CONST_DATASET_PARTITION['Num_of_Data'][id_partition])
            iter_save_data = CONST_DATASET_PARTITION['Init_Index'][id_partition] + percentage_stored_data

            # Load a raw image from a file.
            #image_data = cv2.imread(f'{project_folder}/Data/{CONST_DATASET_NAME}/images/{CONST_PARTITION_DATASET}/Image_{CONST_SCAN_ITERATION:05}.png')
            # Load a label (annotation) from a file.
            #label_data = File_IO.Load(f'{project_folder}/Data/{CONST_DATASET_NAME}/labels/{CONST_PARTITION_DATASET}/Image_{CONST_SCAN_ITERATION:05}', 'txt', ' ')

            print(['train', 'valid', 'text'][id_partition])
            i += 1; percentage_stored_data += 1

    except AssertionError as error:
        print(f'[ERROR] Information: {error}')
        print('[ERROR] The sum of the partition dataset must be 100.')

    print('[INFO] The data augmentation has been successfully completed.')

if __name__ == '__main__':
    sys.exit(main())


