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

def Object_Visibility(name: str, state: bool) -> None:
    """
    Description:
        Function to hide and unhide the visibility of objects.
    
    Args:
        (1) name [string]: Name of the main object.
        (2) state [bool]: Unhide (True) / Hide (False).  
    """
    
    cmd = not state; obj = bpy.data.objects[name]
    
    if Object_Exist(name):
        obj.hide_viewport = cmd; obj.hide_render = cmd
        for obj_i in obj.children:
            obj_i.hide_viewport = cmd; obj_i.hide_render = cmd

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
    """
    Description:
        ...

        {'x': [0.0, 0.0], 'y':, [0.0, 0.0], 'z': [0.0, 0.0]}
    """
        
    def __init__(self) -> None:
        pass

class Object_Cls(object):
    """
    Description:
        ...

        {'x': [0.0, 0.0], 'y':, [0.0, 0.0], 'z': [0.0, 0.0]}
    """
    def __init__(self, name: str, T_0: tp.List[tp.List[float]], limits_position: tp.Tuple[tp.List[float], tp.List[float], tp.List[float]], 
                 limits_rotation: tp.Tuple[tp.List[float], tp.List[float], tp.List[float]], axes_sequence_cfg: str) -> None:
        self.__name = name

        if isinstance(T_0, (list, np.ndarray)):
            self.__T_0 = Transformation.Homogeneous_Transformation_Matrix_Cls(T_0, np.float32)
        else:
            self.__T_0 = T_0

        self.__T = Transformation.Homogeneous_Transformation_Matrix_Cls(T_0.all().copy, np.float32)
        self.__l_p = limits_position
        self.__l_theta = limits_rotation
        self.__axes_sequence_cfg = axes_sequence_cfg

        # ....
        self.Reset()

    @property
    def Name(self):
        self.__name

    @property
    def T_0(self):
        return self.__T_0
    
    @property
    def T(self):
        return self.__T

    def Reset(self) -> None:
        self.__T = Transformation.Homogeneous_Transformation_Matrix_Cls(self.__T_0.all().copy, np.float32)

        Set_Object_Transformation(self.__name, self.__T_0)

    def Visibility(self, state: bool) -> None:
        Object_Visibility(self.__name, state)
        
    def Random(self) -> None:
        p = Transformation.Vector3_Cls(None, np.float32)
        theta = Transformation.Euler_Angle_Cls(None, 'ZYX', np.float32)

        for i, (l_p_i, l_theta_i) in enumerate(zip(self.__l_p, self.__l_theta)):
            if l_p_i != None:
                p[i] = np.random.uniform(l_p_i[0], l_p_i[1])

            if l_theta_i != None:
                theta[i] = np.random.uniform(l_theta_i[0], l_theta_i[1])

        self.__T = Get_Transformation_Matrix(p, theta, self.__axes_sequence_cfg)

        Set_Object_Transformation(self.__name, self.__T)



