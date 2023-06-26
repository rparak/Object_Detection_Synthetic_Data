# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('..')
# OS (Operating system interfaces)
import os
# Time (Time access and conversions)
import time
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
# The ID of the object to be augmented.
#   ID{0} = 'T_Joint'
#   ID{1} = 'Metal_Blank'
#   ID{'B'} = 'Background'
CONST_OBJECT_ID = 0
# The identification number of the dataset type.
CONST_DATASET_TYPE = 1
# Name of the dataset.
CONST_DATASET_NAME = f'Dataset_Type_{CONST_DATASET_TYPE}'
# Number of augmented data to be generated.
CONST_NUM_OF_GEN_DATA = 285
# Explanation of the dictionary of dataset partitions.
#   'Percentage': Partition the dataset into training, validation, and test sets in percentages.
#                 Note:
#                   The sum of the values in the partitions must equal 100.
#   'Init_Index': The initial index (iteration) of the augmented data in each partition.
#                 Note:
#                   The value of the initial index is equal to the last image/label value 
#                   in the dataset + 1.
#   'Num_of_Data': Number of data in each partitions.
CONST_DATASET_PARTITION = {'Percentage': [80, 20, 0], 'Init_Index': [13, 244, 0], 'Num_of_Data': [12, 3, 0]}

def main():
    """
    Descriptioon:
        The main program that generates (augments) data from a small image dataset. The program includes image and label 
        augmentations.

        A small dataset for the augmentation can be found here:
            ../Data/Dataset_Type_0/..

        Note:
            The principle of augmented data can be changed by modifying the declaration 
            of transformation.

                A.Compose([..], bbox_params=A.BboxParams(..))
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Object_Detection_Synthetic_Data')[0] + 'Object_Detection_Synthetic_Data'

    try:
        assert np.sum(CONST_DATASET_PARTITION['Percentage']) == 100

        # # Generates data up to the desired maximum number of iterations, which is given by the constant {CONST_NUM_OF_GEN_DATA}.
        i = 0; id_partition = 0; percentage_stored_data = 0
        while CONST_NUM_OF_GEN_DATA > i:
            # Start the timer.
            t_0 = time.time()

            # Calculate the current percentage of the data.
            if percentage_stored_data == (CONST_NUM_OF_GEN_DATA * (CONST_DATASET_PARTITION['Percentage'])[id_partition]/100):
                id_partition += 1; percentage_stored_data = 0
                if CONST_DATASET_PARTITION['Percentage'][id_partition] == 0:
                    id_partition += 1

            # Describe the current iteration for loading/saving data (image/label).
            iter_load_data = (percentage_stored_data % CONST_DATASET_PARTITION['Num_of_Data'][id_partition]) \
                           + (CONST_DATASET_PARTITION['Init_Index'][id_partition] - CONST_DATASET_PARTITION['Num_of_Data'][id_partition])
            iter_save_data = CONST_DATASET_PARTITION['Init_Index'][id_partition] + percentage_stored_data

            # Partition ID as a string.
            partition_name = ['train', 'valid', 'text'][id_partition]
            # Create a file path to load/save the data.
            file_path_load_data = f'{project_folder}/Data/{CONST_DATASET_NAME}'
            file_path_save_data = f'{project_folder}/Data/{CONST_DATASET_NAME}'

            # Load a raw image from a file.
            image_data = cv2.imread(f'{file_path_load_data}/images/{partition_name}/Object_ID_{CONST_OBJECT_ID}_{iter_load_data:05}.png')

            if CONST_OBJECT_ID != 'B':
                # Load a label (annotation) from a file.
                label_data = File_IO.Load(f'{file_path_load_data}/labels/{partition_name}/Object_ID_{CONST_OBJECT_ID}_{iter_load_data:05}', 'txt', ' ')[0]
        
                # The transformation declaration used to augment the image/bounding box.
                #   More information on the transformation can be found here:
                #       http://albumentations.ai  
                transformation = A.Compose([A.Affine(translate_px={'x': (-10, 10), 'y': (-10, 10)}, p = 0.75),
                                            A.ColorJitter(brightness=(0.25, 1.5), contrast=(0.25, 1.5), saturation=(0.1, 1.0), 
                                                          always_apply=True),
                                            A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(0.01, 1.0), p = 0.5),
                                            A.RandomResizedCrop(height= 1544, width = 2064, scale = (0.95, 1.0), p = 0.5)], 
                                            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
            
                # Argumentation of both the image and the bounding box.
                augmented = transformation(image = image_data, bboxes = [label_data[1::]], class_labels=[label_data[0]])
    
                # Save the label data (bounding box) to a file.
                File_IO.Save(f'{file_path_save_data}/labels/{partition_name}/Object_ID_{CONST_OBJECT_ID}_{iter_save_data:05}', 
                             np.hstack((int(augmented['class_labels'][0]), augmented['bboxes'][0])), 'txt', ' ')
            else:
                transformation = A.Compose([A.ColorJitter(brightness=(0.25, 1.5), contrast=(0.25, 1.5), saturation=(0.1, 1.0), 
                                                          always_apply=True),
                                            A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(0.01, 1.0), p = 0.5)])
                augmented = transformation(image = image_data)

                # Save the empty data to a file.
                with open(f'{file_path_save_data}/labels/{partition_name}/Object_ID_{CONST_OBJECT_ID}_{iter_save_data:05}.txt', 'a+') as f:
                    pass
                f.close()

            # Save the image to a file.
            cv2.imwrite(f'{file_path_save_data}/images/{partition_name}/Object_ID_{CONST_OBJECT_ID}_{iter_save_data:05}.png', augmented['image'])

            # Display information.
            print(f'[INFO] The data in iteration {int(i)} was successfully saved to the folder {file_path_save_data}.')
            print(f'[INFO]  - Image: /images/{partition_name}/Object_ID_{CONST_OBJECT_ID}_{iter_save_data:05}.png')
            print(f'[INFO]  - Label: /labels/{partition_name}/Object_ID_{CONST_OBJECT_ID}_{iter_save_data:05}.txt')
            print(f'[INFO] Time: {(time.time() - t_0):0.05f} in seconds.')

            i += 1; percentage_stored_data += 1

    except AssertionError as error:
        print(f'[ERROR] Information: {error}')
        print('[ERROR] The sum of the partition dataset must be 100.')

    print('[INFO] The data augmentation has been successfully completed.')

if __name__ == '__main__':
    sys.exit(main())


