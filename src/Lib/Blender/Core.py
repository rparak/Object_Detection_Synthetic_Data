# BPY (Blender as a python) [pip3 install bpy]
import bpy
# System (Default)
import sys
sys.path.append('..')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# Custom Library:
#   ../Lib/Transformation/Core
import Lib.Transformation.Core as Transformation

#import cv2

# Focal length of spherical concave mirror is .. f = alpha / 2
# Focal length: f = 

def Object_Exist(name: str) -> bool:
    """
    Description:
        Check if the object exists within the scene.
        
    Args:
        (1) name [string]: Object name.
    Returns:
        (1) parameter [bool]: 'True' if it exists, otherwise 'False'.
    """
    
    return True if bpy.context.scene.objects.get(name) else False

def Deselect_All() -> None:
    """
    Description:
        Deselect all objects in the current scene.
    """
    
    for obj in bpy.context.selected_objects:
        bpy.data.objects[obj.name].select_set(False)

def Set_Object_Transformation(name: str, T: tp.List[tp.List[float]]) -> None:
    """
    Description:
        Set the object transformation.
    Args:
        (1) name [string]: Name of the main object.
        (2) T [Matrix<float> 4x4]: Homogeneous transformation matrix (access to location, rotation and scale).
    """

    if isinstance(T, (list, np.ndarray)):
        T = Transformation.Homogeneous_Transformation_Matrix_Cls(T, np.float32)
    
    bpy.data.objects[name].matrix_basis = T.Transpose().all()

def Get_Transformation_Matrix(location: tp.List[float], rotation: tp.List[float], axes_sequence_cfg: str) -> tp.List[float]:       
    """
    Description:
        Obtain a homogeneous transformation matrix from the specified input parameters.

    Args:
        (1) location [Vector<float> 1x3]: Direction vector (x, y, z).
        (2) rotation [Vector<float> 1x3, 1x4]: Angle of rotation defined in the specified form (axes sequence 
                                               configuration): Euler Angles, Quaternions, etc. 
        (3) axes_sequence_cfg [string]: Rotation axis sequence configuration (e.g. 'ZYX', 'QUATERNION', etc.)

    Returns:
        (1) parameter [Matrix<cls_data_type> 4x4]: Homogeneous transformation matrix {T} transformed according to the input 
                                                   parameters {location, rotation}.
    """

    T = Transformation.Homogeneous_Transformation_Matrix_Cls(None, np.float32)

    return T.Translation(location).Rotation(rotation, axes_sequence_cfg)

class Camera_Cls(object):
    def __init__(self) -> None:
        pass

class Object_Cls(object):
    def __init__(self) -> None:
        pass



