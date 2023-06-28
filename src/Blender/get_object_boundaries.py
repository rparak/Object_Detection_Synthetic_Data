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
        $ cd Documents/GitHub/Object_Detection_Synthetic_Data/Blender
        $ blender Gen_Synthetic_Data.blend
"""

"""
Description:
    Initialization of constants.
"""
# The ID of the object to be generated.
#   ID{0} = 'T_Joint'
#   ID{1} = 'Metal_Blank'
CONST_OBJECT_ID = 0
# Number of data to be generated.
#   Note:
#       More data could be useful for better accuracy.
CONST_NUM_OF_DATA = 10000

def main():
    """
    Description:
        A program to get the boundaries of an object (bounding box). More precisely the area of the rectangle.

        Rectangle Area:
            A = w * h,

            where w is the width and h is the height of the 2D coordinates of the bounding box.

        Boundaries (limits):
            A_{-}, A_{+}
    """

    # Deselect all objects in the current scene.
    Lib.Blender.Utilities.Deselect_All()

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

    # Generates data up to the desired maximum number of iterations, which is given by the constant {CONST_NUM_OF_DATA}.
    i = 0; A = []
    while CONST_NUM_OF_DATA > i:
        # Generate a random position of the object.
        Object_Cls.Random()

        # Get the 2D coordinates of the bounding box from the rendered object scanned by the camera.
        bounding_box_2d = Lib.Utilities.General.Get_2D_Coordinates_Bounding_Box(Object_Cls.Vertices, Camera_Cls.P(), 
                                                                                Lib.Parameters.Camera.PhoXi_Scanner_M_Str.Resolution, 'YOLO')
        
        
        # Get the area of the rectangle.
        A.append(bounding_box_2d['width'] * bounding_box_2d['height'])
        i += 1
        
    # Display information.
    print('[INFO] The average area of a rectangle:')
    print(f'[INFO]   A = {np.sum(A)/len(A)}')
    print('[INFO] Boundaries (limits):')
    print(f'[INFO]   A_[-] = {np.min(A)}')
    print(f'[INFO]   A_[+] = {np.max(A)}')

    # Disable (turn off) visibility of the object.
    Object_Cls.Visibility(False)

if __name__ == '__main__':
    main()