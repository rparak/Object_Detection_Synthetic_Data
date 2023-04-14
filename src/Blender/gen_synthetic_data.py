# BPY (Blender as a python) [pip3 install bpy]
import bpy
# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../' + 'src')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Custom Library:
#   ../Lib/Blender/Core & Utilities
import Lib.Blender.Core
import Lib.Blender.Utilities
#   ../Lib/Parameters/Camera & Object
import Lib.Parameters.Camera
import Lib.Parameters.Object

"""
Description:
    Initialization of constants.
"""
# Number of synthetic data to be generated.
CONST_NUM_OF_GEN_DATA = 10
# The object to be scanned.
CONST_ID_SCANNED_OBJECT = 0

def main():
    # Deselect all objects in the current scene.
    Lib.Blender.Utilities.Deselect_All()

    # ...
    Camera_Cls = Lib.Blender.Core.Camera_Cls('Camera', Lib.Parameters.Camera.PhoXi_Scanner_M_Str, 'PNG')
    # ...
    #   ...
    T_Joint_001_Cls = Lib.Blender.Core.Object_Cls(Lib.Parameters.Object.T_Joint_001_Str, 'ZYX')
    if Lib.Parameters.Object.T_Joint_001_Str.Id == CONST_ID_SCANNED_OBJECT:
        T_Joint_001_Cls.Visibility(True)
    else:
        T_Joint_001_Cls.Visibility(False)
    #   ...
    Metal_Blank_001_Cls = Lib.Blender.Core.Object_Cls(Lib.Parameters.Object.Metal_Blank_001_Str, 'ZYX')
    if Lib.Parameters.Object.Metal_Blank_001_Str.Id == CONST_ID_SCANNED_OBJECT:
        Metal_Blank_001_Cls.Visibility(True)
    else:
        Metal_Blank_001_Cls.Visibility(False)
        
    # save data ...
    i = 9
    Lib.Blender.Utilities.Save_Synthetic_Data('../Data/Train/', f'Test_Name_{i:07}', 0, [0.504347,0.442419,0.087728,0.089276], 
                                              'txt', 'png')

if __name__ == '__main__':
    main()