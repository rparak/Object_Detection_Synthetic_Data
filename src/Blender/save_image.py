# BPY (Blender as a python) [pip3 install bpy]
import bpy
# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../' + 'src')
# OS (Operating system interfaces)
import os
# Custom Library:
#   ../Lib/Blender/Core & Utilities
import Lib.Blender.Core
import Lib.Blender.Utilities
#   ../Lib/Parameters/Camera & Object
import Lib.Parameters.Camera
import Lib.Parameters.Object

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
#   ID{0} = 'T_Joint'
#   ID{1} = 'Metal_Blank'
CONST_OBJECT_ID = 0

def main():
    """
    Description:
        A simple script to save an image from the camera to the Desktop folder.
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Object_Detection_Synthetic_Data')[0] + 'Object_Detection_Synthetic_Data'
    
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
    Object_Cls.Visibility(False)

    # Generate random camera and lighting properties.
    Camera_Cls.Random()

    # Save the image to the file.
    bpy.context.scene.render.filepath = f'{project_folder}/images/Object_ID_{Object_Str.Id}_{Object_Str.Name}.png'
    bpy.ops.render.render(animation=False, write_still=True)
    
    # Disable (turn off) visibility of the object.
    Object_Cls.Visibility(False)

if __name__ == '__main__':
    main()
