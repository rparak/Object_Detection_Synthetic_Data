# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../' + 'src')
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
    Initialization of constants.
"""
# Number of synthetic data to be generated.
CONST_NUM_OF_GEN_DATA = 10
# Initial index for data generation.
CONST_INIT_INDEX = 0

def main():
    # Deselect all objects in the current scene.
    Lib.Blender.Utilities.Deselect_All()

    # The object to be scanned.
    #   'T_Joint': Lib.Parameters.Object.T_Joint_001_Str
    #   'Metal_Blank': Lib.Parameters.Object.Metal_Blank_001_Str
    Object_Str = Lib.Parameters.Object.T_Joint_001_Str
    
    # ...
    Camera_Cls = Lib.Blender.Core.Camera_Cls('Camera', Lib.Parameters.Camera.PhoXi_Scanner_M_Str, 'PNG')
    # ...
    Object_Cls = Lib.Blender.Core.Object_Cls(Lib.Parameters.Object.T_Joint_001_Str, 'ZYX')
    Object_Cls.Visibility(True)
    
    i = 0
    while CONST_NUM_OF_GEN_DATA < i:
        # ...
        Object_Cls.Random()

        # ...
        bounding_box_2d = Lib.Utilities.General.Get_2D_Coordinates_Bounding_Box(Object_Cls.Vertices, Camera_Cls.P(), 
                                                                                Lib.Parameters.Camera.PhoXi_Scanner_M_Str.Resolution, 'YOLO')
        
        # ...
        Lib.Blender.Utilities.Save_Synthetic_Data('../Data/Train', f'{CONST_INIT_INDEX + (i + 1):07}', Object_Str.Id, list(bounding_box_2d.values()), 
                                                  'txt', 'png')
        i += 1

    Object_Cls.Visibility(False)

if __name__ == '__main__':
    main()