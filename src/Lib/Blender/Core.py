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
#   ../Lib/Parameters/Camera & Object
import Lib.Parameters.Camera
import Lib.Parameters.Object

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
    
    bpy.data.objects[name].matrix_basis = T.Transpose().all().copy()

def Get_Transformation_Matrix(position: tp.List[float], rotation: tp.List[float], axes_sequence_cfg: str) -> tp.List[float]:       
    """
    Description:
        Obtain a homogeneous transformation matrix from the specified input parameters.

    Args:
        (1) position [Vector<float> 1x3]: Direction vector (x, y, z).
        (2) rotation [Vector<float> 1x3, 1x4]: Angle of rotation defined in the specified form (axes sequence 
                                               configuration): Euler Angles, Quaternions, etc. 
        (3) axes_sequence_cfg [string]: Rotation axis sequence configuration (e.g. 'ZYX', 'QUATERNION', etc.)

    Returns:
        (1) parameter [Matrix<cls_data_type> 4x4]: Homogeneous transformation matrix {T} transformed according to the input 
                                                   parameters {position, rotation}.
    """

    T = Transformation.Homogeneous_Transformation_Matrix_Cls(None, np.float32)

    return T.Translation(position).Rotation(rotation, axes_sequence_cfg)

class Camera_Cls(object):
    """
    Description:
        ...
        (1) name [string]: Object name.
    """
        
    def __init__(self, name: str, Cam_Param_Str: Lib.Parameters.Camera.Camera_Parameters_Str, image_format: str = 'PNG') -> None:
        self.__name = name
        self.__Cam_Param_Str = Cam_Param_Str
        self.__image_format = image_format

        self.__Set_Camera_Parameters()

    # Camera Parameters vs Properties
    def __Set_Camera_Parameters(self) -> None:
        # ...
        T_Cam = Get_Transformation_Matrix(self.__Cam_Param_Str.Position.all(), 
                                          self.__Cam_Param_Str.Rotation.all(), 'XYZ')
        Set_Object_Transformation(self.__name, T_Cam)
        # Adjust the width or height of the sensor depending on the resolution of the image.
        bpy.data.cameras[self.__name].sensor_fit = 'AUTO'
        # ...
        bpy.context.scene.render.resolution_x = self.__Cam_Param_Str.Resolution['x']
        bpy.context.scene.render.resolution_y = self.__Cam_Param_Str.Resolution['y']
        # Set the projection of the camera
        bpy.data.cameras[self.__name].type = self.__Cam_Param_Str.Type
        bpy.data.cameras[self.__name].lens_unit = 'MILLIMETERS'
        bpy.data.cameras[self.__name].lens = self.__Cam_Param_Str.f
        # Color Management
        bpy.context.scene.view_settings.gamma = self.__Cam_Param_Str.Gamma
        # Output image settings (8-bit, 16-bit)
        bpy.context.scene.render.image_settings.color_depth = '8'
        # BW vs RGBA
        if self.__Cam_Param_Str.Spectrum == 'Monochrome':
            bpy.context.scene.render.image_settings.color_mode = 'BW'
        elif self.__Cam_Param_Str.Spectrum == 'Color':
            bpy.context.scene.render.image_settings.color_mode = 'RGBA'

        # ...
        bpy.context.scene.render.image_settings.file_format = self.__image_format

        # Update the scene.
        bpy.context.view_layer.update()

    # https://learnopencv.com/camera-calibration-using-opencv/
    # https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
    # https://visp-doc.inria.fr/doxygen/visp-3.4.0/tutorial-tracking-mb-generic-rgbd-Blender.html
    # https://github.com/vsitzmann/shapenet_renderer/blob/master/util.py
    # https://mcarletti.github.io/articles/blenderintrinsicparams/
    # https://docs.opencv.org/3.4.3/d9/d0c/group__calib3d.html#details

    def K(self) -> tp.List[tp.List[float]]:
        try:
            assert bpy.data.cameras[self.__name].sensor_fit == 'AUTO'

            # intrinsic matrix
            # ...
            alpha_u = (self.__Cam_Param_Str.f * self.__Cam_Param_Str.Resolution['x']) / bpy.data.cameras[self.__name].sensor_width
            alpha_v = (self.__Cam_Param_Str.f * self.__Cam_Param_Str.Resolution['y']) / bpy.data.cameras[self.__name].sensor_height
            # Only use rectangular pixels ...
            s = 0.0
            # ...
            u_0 = self.__Cam_Param_Str.Resolution['x'] / 2.0
            v_0 = self.__Cam_Param_Str.Resolution['y'] / 2.0

            return Transformation.Homogeneous_Transformation_Matrix_Cls([[alpha_u,       s, u_0, 0.0],
                                                                         [    0.0, alpha_v, v_0, 0.0],
                                                                         [    0.0,     0.0, 1.0, 0.0],
                                                                         [    0.0,     0.0, 0.0, 1.0]], np.float32).R
        
        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
            print('[ERROR] Incorrectly set method to fit the image and field of view angle inside the sensor.')
            print(f'[ERROR] The method must be set to AUTO. Not to {bpy.data.cameras[self.__name].sensor_fit}')

    def RT(self):
        # extrinsic matrix (R | T)
        pass

    def P(self):
        return self.K() @ self.RT()

    def Save_Data(self, image_properties) -> None:
        # image_properties = {'Path': .., 'Name': ..}
        # label_properties = {'Path': .., 'Name': .., 'Data': ..}
        
        # ...
        img_path = image_properties['Path']; img_name = image_properties['Name']
        #   ...
        bpy.context.scene.render.filepath = f'{img_path}/{img_name}.{self.__image_format.lower()}'
        bpy.ops.render.render(write_still=True)

class Object_Cls(object):
    """
    Description:
        ...
    """
    def __init__(self, Obj_Param_Str: Lib.Parameters.Object.Object_Parameters_Str, axes_sequence_cfg: str) -> None:
        self.__Obj_Param_Str = Obj_Param_Str
        self.__T = Transformation.Homogeneous_Transformation_Matrix_Cls(self.__Obj_Param_Str.T.all().copy(), np.float32)
        self.__axes_sequence_cfg = axes_sequence_cfg

        # ....
        self.Reset()

    @property
    def Name(self):
        return self.__Obj_Param_Str.Name

    @property
    def Id(self):
        return self.__Obj_Param_Str.Id
    
    @property
    def T_0(self):
        return self.__Obj_Param_Str.T
    
    @property
    def T(self):
        return self.__T

    def Reset(self) -> None:
        self.__T = Transformation.Homogeneous_Transformation_Matrix_Cls(self.__Obj_Param_Str.T.all().copy(), np.float32)

        Set_Object_Transformation(self.__Obj_Param_Str.Name, self.__Obj_Param_Str.T)

        # Update the scene.
        bpy.context.view_layer.update()

    def Visibility(self, state: bool) -> None:
        Object_Visibility(self.__Obj_Param_Str.Name, state)
        
    def Random(self) -> None:
        p = np.zeros(3, np.float32); theta = p.copy()
        for i, (l_p_i, l_theta_i) in enumerate(zip(self.__Obj_Param_Str.Limit.Position.items(), 
                                                   self.__Obj_Param_Str.Limit.Rotation.items())):
            if l_p_i[1] != None:
                p[i] = np.random.uniform(l_p_i[1][0], l_p_i[1][1])

            if l_theta_i[1] != None:
                theta[i] = np.random.uniform(l_theta_i[1][0], l_theta_i[1][1])

        self.__T = self.__Obj_Param_Str.T @ Get_Transformation_Matrix(p, theta, self.__axes_sequence_cfg)

        Set_Object_Transformation(self.__Obj_Param_Str.Name, self.__T)

        # Update the scene.
        bpy.context.view_layer.update()



