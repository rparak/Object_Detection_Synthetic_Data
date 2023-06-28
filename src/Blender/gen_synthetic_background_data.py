# BPY (Blender as a python) [pip3 install bpy]
import bpy
# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../' + 'src')
# OS (Operating system interfaces)
import os
# Time (Time access and conversions)
import time
# Custom Library:
#   ../Lib/Blender/Core & Utilities
import Lib.Blender.Core
import Lib.Blender.Utilities
#   ../Lib/Parameters/Camera
import Lib.Parameters.Camera

"""
Description:
    Open Gen_Synthetic_Data.blend from the Blender folder and copy + paste this script and run it.

    Terminal:
        $ cd Documents/GitHub/Object_Detection_Synthetic_Data/Blender
        $ blender Gen_Synthetic_Data.blend
"""

"""
Description:
    Initialization of constants.
"""
# The ID of the object to be scanned.
#   ID{'B'} = 'Background'
CONST_OBJECT_ID = 'B'
# The identification number of the dataset type.
CONST_DATASET_TYPE = 3
# Number of synthetic data to be generated.
CONST_NUM_OF_GEN_DATA = 20
# Name of the dataset.
CONST_DATASET_NAME = f'Dataset_Type_{CONST_DATASET_TYPE}'

def main():
    """
    Description:
        The main program to generate data of the background from Blender.
    """

    # Deselect all objects in the current scene.
    Lib.Blender.Utilities.Deselect_All()

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Object_Detection_Synthetic_Data')[0] + 'Object_Detection_Synthetic_Data'
    
    # Initialize the camera to scan an object in the scene.
    #   The main parameters of the camera can be found at: ../Lib/Parameters/Camera
    Camera_Cls = Lib.Blender.Core.Camera_Cls('Camera', Lib.Parameters.Camera.PhoXi_Scanner_M_Str, 'PNG')

    # Generates data up to the desired maximum number of iterations, which is given by the constant {CONST_NUM_OF_GEN_DATA}.
    i = 0
    while CONST_NUM_OF_GEN_DATA > i:
        # Generate random camera and lighting properties.
        Camera_Cls.Random()
    
        # Start the timer.
        t_0 = time.time()

        # Save the image to the file.
        bpy.context.scene.render.filepath = f'{project_folder}/Data/{CONST_DATASET_NAME}/images/train/Object_ID_{CONST_OBJECT_ID}_{(i+1):05}.png'
        bpy.ops.render.render(animation=False, write_still=True)

        # Save the empty data to a file.
        with open(f'{project_folder}/Data/{CONST_DATASET_NAME}/labels/train/Object_ID_{CONST_OBJECT_ID}_{(i+1):05}.txt', 'a+') as f:
            pass
        f.close()

        # Display information.
        print(f'[INFO] The data in iteration {int(i)} was successfully saved to the folder {project_folder}/Data/{CONST_DATASET_NAME}/.')
        print(f'[INFO]  - Image: /images/train/Object_ID_{CONST_OBJECT_ID}_{(i+1):05}.png')
        print(f'[INFO]  - Label: /labels/train/Object_ID_{CONST_OBJECT_ID}_{(i+1):05}.txt')
        print(f'[INFO] Time: {(time.time() - t_0):0.05f} in seconds.')
        i += 1

    print('[INFO] The data generation has been successfully completed.')
if __name__ == '__main__':
    main()