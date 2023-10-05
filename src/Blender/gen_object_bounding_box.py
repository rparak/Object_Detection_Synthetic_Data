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
#   ../Lib/Blender/Core/Utilities
import Lib.Blender.Utilities
#   ../Lib/Transformation/Core
import Lib.Transformation.Core as Transformation
#   ../Lib/Utilities/General
import Lib.Utilities.General

"""
Description:
    Open Object_Bounding_Box.blend from the Blender folder and copy + paste this script and run it.

    Terminal:
        $ cd Documents/GitHub/Object_Detection_Synthetic_Data/Blender
        $ blender Object_Bounding_Box.blend
"""

def main():
    """
    Description:
        The main program to generate a 3D bounding box for each object in the scene.

        Find and display the main bounding box parameters relevant to the main object structure.
            Object_Parameters_Str() in ../Lib/Parameters/Object.py
    """
    
    # Deselect all objects in the current scene.
    Lib.Blender.Utilities.Deselect_All()
    
    # The identity homogeneous transformation matrix. Zero position.
    T_0 = Transformation.Homogeneous_Transformation_Matrix_Cls(None, np.float64)
    
    for obj in bpy.data.objects:
        # Reset the position and rotation of the object in the scene.
        Lib.Blender.Utilities.Set_Object_Transformation(obj.name, T_0)

        # Set the rotation mode to "ZYX" notation (ax. sequence configuration).
        bpy.data.objects[obj.name].rotation_mode = 'ZYX'
        
        # Calculate the bounding box parameters from the vertices of the object.
        #   Get the minimum and maximum values of the X, Y, Z vertices of the object.
        (min_verts, max_verts) = Lib.Utilities.General.Get_Min_Max(np.array(Lib.Blender.Utilities.Get_Vertices_From_Object(obj.name), dtype=np.float64))
        #   Obtain the main parameters of the bounding box.
        origin = []; size = []
        for _, (min_v_i, max_v_i) in enumerate(zip(min_verts, max_verts)):
            origin.append((max_v_i + min_v_i)/2.0); size.append(np.abs(max_v_i - min_v_i))
            
        # Convery list to numpy array.
        origin = np.array(origin, np.float64); size = np.array(size, np.float64)
        
        # Display the results of the calculation.
        print(f'[INFO] Object name: {obj.name}')
        print('[INFO] Bounding Box Parameters:')
        print(f'[INFO]  - Origin: {origin}')
        print(f'[INFO]  - Size: {size}')
        
        # Create an object (bounding box) from the calculated parameters.
        #   Parametres of the created object.
        box_properties = {'transformation': {'Size': 1.0, 'Scale': size, 'Location': [0.0, 0.0, 0.0]}, 
                          'material': {'RGBA': [0.8,0.8,0.8,1.0], 'alpha': 0.1}}
                          
        # Create a primitive three-dimensional object (bounding box) with additional properties.
        Lib.Blender.Utilities.Create_Primitive('Cube', f'{obj.name}_Bounding_Box', box_properties)
        bpy.data.objects[f'{obj.name}_Bounding_Box'].rotation_mode = 'ZYX'
        print('[INFO] The bounding box was successfully created!')
        
        # Set the origin of the object (bounding box).
        Lib.Blender.Utilities.Set_Object_Origin(f'{obj.name}_Bounding_Box', (-1) * origin)
        
        # Reset the position and rotation of the object (bounding box) in the scene.
        Lib.Blender.Utilities.Set_Object_Transformation(f'{obj.name}_Bounding_Box', T_0)

if __name__ == '__main__':
    main()
