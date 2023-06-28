# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../' + 'src')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# OS (Operating system interfaces)
import os
# Custom Library:
#   ../Lib/Blender/Core & Utilities
import Lib.Blender.Core
import Lib.Blender.Utilities
#   ../Lib/Parameters/Camera & Object
import Lib.Parameters.Camera
import Lib.Parameters.Object
#   ../Lib/Utilities/General
import Lib.Utilities.General

"""
Description:
    Open Gen_Synthetic_Data.blend from the Blender folder and copy + paste this script and run it.

    Terminal:
        $ cd Documents/GitHub/Blender_Synthetic_Data/Blender
        $ blender Gen_Synthetic_Data.blend
"""

"""
Description:
    Initialization of constants.
"""
# The ID of the object to be generated.
#   ID{0} = 'T_Joint'
#   ID{1} = 'Metal_Blank'
#   ID{'B'} = 'Background'
CONST_OBJECT_ID = 0
# The identification number of the dataset type.
CONST_DATASET_TYPE = 3
# Number of synthetic data to be generated.
CONST_NUM_OF_GEN_DATA = 240
# Name of the dataset.
CONST_DATASET_NAME = f'Dataset_Type_{CONST_DATASET_TYPE}'
# Partition the dataset into training, validation, and test sets in percentages.
#   Note:
#       The sum of the values in the partitions must equal 100.
CONST_PARTITION_DATASET = {'train': 100, 'valid': 0, 'test': 0}
# Initial index (iteration) for data generation.
#   0 - Data storage starts from 1 (Name_001, etc.)
#   10 - Data storage start from 11 (Name_011, etc.)
CONST_INIT_INDEX = 0

def main():
    """
    Description:
        The main program that generates data from Blender. The main parameters of the generation can be found 
        in the constants at the top.

        Note 1:
            Synthetic data consists of an image and a corresponding label, which is defined by a bounding box 
            around the scanned object.

        Note 2:
            The data generation process only applies to the 2D area of view.
    """

    # Deselect all objects in the current scene.
    Lib.Blender.Utilities.Deselect_All()

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Object_Detection_Synthetic_Data')[0] + 'Object_Detection_Synthetic_Data'
    
    # Select the structure of the scanned object.
    Object_Str = [Lib.Parameters.Object.T_Joint_001_Str, 
                  Lib.Parameters.Object.Metal_Blank_001_Str][CONST_OBJECT_ID]
    
    # Initialize the camera to scan an object in the scene.
    #   The main parameters of the camera can be found at: ../Lib/Parameters/Camera
    Camera_Cls = Lib.Blender.Core.Camera_Cls('Camera', Lib.Parameters.Camera.PhoXi_Scanner_M_Str, 'PNG')
    # Initialize the object to be scanned by the camera.
    #   The main parameters of the object can be found at: ../Lib/Parameters/Object
    Object_Cls = Lib.Blender.Core.Object_Cls(Object_Str, 'ZYX')
    # Enable (turn on) visibility of the object.
    Object_Cls.Visibility(True)
    
    try:
        assert np.sum(list(CONST_PARTITION_DATASET.values())) == 100

        # Generates data up to the desired maximum number of iterations, which is given by the constant {CONST_NUM_OF_GEN_DATA}.
        i = 0; id_partition = 0; percentage_stored_data = 0
        while CONST_NUM_OF_GEN_DATA > i:
            # Generate random camera and lighting properties.
            Camera_Cls.Random()

            # Generate a random position of the object.
            Object_Cls.Random()

            # Get the 2D coordinates of the bounding box from the rendered object scanned by the camera.
            bounding_box_2d = Lib.Utilities.General.Get_2D_Coordinates_Bounding_Box(Object_Cls.Vertices, Camera_Cls.P(), 
                                                                                    Lib.Parameters.Camera.PhoXi_Scanner_M_Str.Resolution, 'YOLO')
            
            # Calculate the current percentage of stored data.
            if percentage_stored_data == (CONST_NUM_OF_GEN_DATA * (list(CONST_PARTITION_DATASET.values())[id_partition]/100)):
                id_partition += 1; percentage_stored_data = 0
                if list(CONST_PARTITION_DATASET.values())[id_partition] == 0:
                    id_partition += 1

            # Get the name of the partition where the data will be stored.
            partition_name = list(CONST_PARTITION_DATASET.keys())[id_partition]

            # Save the image with the corresponding label.
            Lib.Blender.Utilities.Save_Synthetic_Data(f'{project_folder}/Data/{CONST_DATASET_NAME}/', partition_name, f'{CONST_INIT_INDEX + (i + 1):05}', 
                                                      Object_Str.Id, list(bounding_box_2d.values()), 'txt', 'png')
            i += 1; percentage_stored_data += 1
            
    except AssertionError as error:
        print(f'[ERROR] Information: {error}')
        print('[ERROR] The sum of the partition dataset must be 100.')

    print('[INFO] The data generation has been successfully completed.')
    
    # Disable (turn off) visibility of the object.
    Object_Cls.Visibility(False)

if __name__ == '__main__':
    main()