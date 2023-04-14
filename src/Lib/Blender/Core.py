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
#   ../Lib/Blender/Core & Utilities
import Lib.Blender.Core
import Lib.Blender.Utilities
#   ../Lib/Parameters/Camera & Object
import Lib.Parameters.Camera
import Lib.Parameters.Object
#   ../Lib/Transformation/Core
import Lib.Transformation.Core as Transformation

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
        Lib.Blender.Utilities.Set_Object_Transformation(self.__name, self.__Cam_Param_Str.T)

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

    def K(self) -> tp.List[tp.List[float]]:
        try:
            assert bpy.data.cameras[self.__name].sensor_fit == 'AUTO'

            # intrinsic matrix
            # ...
            alpha_u = (self.__Cam_Param_Str.f * self.__Cam_Param_Str.Resolution['x']) / bpy.data.cameras[self.__name].sensor_width
            alpha_v = alpha_u
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

    def R_t(self):
        # extrinsic matrix (R | t)
        # obj.matrix_world.normalized().inverted()
        #R_n = np.array(R_bcam2cv) @ np.array(cam.matrix_world.inverted())[:3, :3]
        #t_n = np.array(R_bcam2cv) @ ((-1) * np.array(cam.matrix_world.inverted())[:3, :3] @ location)

        R_tmp = np.array([[1.0, 0.0,  0.0],
                          [0.0, 1.0,  0.0],
                          [0.0, 0.0, -1.0]], dtype=np.float32)

        R = R_tmp @ self.__Cam_Param_Str.T.Transpose().R
        t = R_tmp @ ((-1) * self.__Cam_Param_Str.T.Transpose().R @ self.__Cam_Param_Str.T.p.all())

        return np.hstack((R, t.reshape(3, 1)))

    def P(self):
        # K x [R | t]
        return self.K() @ self.R_t()

    def Save_Data(self, image_properties) -> None:
        # Maybe save data of the camera, and of the object as well. Means separately.
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
        try:
            assert Lib.Blender.Utilities.Object_Exist(Obj_Param_Str.Name) == True

            self.__Obj_Param_Str = Obj_Param_Str
            self.__T = Transformation.Homogeneous_Transformation_Matrix_Cls(None, np.float32)
            self.__axes_sequence_cfg = axes_sequence_cfg

            self.__Bounding_Box = {'Centroid': self.__T.p.all(), 'Size': self.__Obj_Param_Str.Bounding_Box.Size, 
                                'Vertices': self.__Obj_Param_Str.Bounding_Box.Vertices.copy()}

            # test something
            Lib.Blender.Utilities.Set_Object_Transformation(self.__Obj_Param_Str.Name, Transformation.Homogeneous_Transformation_Matrix_Cls(None, np.float32))
            self.__Update()

            # print(f'[INFO] An object named <{self.__Obj_Param_Str.Name}_Bounding_Box> does not exist in the current scene.')

            # ....
            self.Reset()
        
        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
            print(f'[INFO] An object named <{Obj_Param_Str.Name}> does not exist in the current scene.')

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
    
    @property
    def Bounding_Box(self):
        """
        Description:
            Get the main parameters of the object's bounding box.
        """

        # Oriented Bounding Box (OBB) transformation according to the homogeneous transformation matrix of the object.
        q = self.__T.Get_Rotation('QUATERNION'); p = self.__T.p.all()
        for i, point_i in enumerate(self.__Obj_Param_Str.Bounding_Box.Vertices):
            self.__Bounding_Box['Vertices'][i, :] = q.Rotate(Transformation.Vector3_Cls(point_i, np.float32)).all() + p

        # The center of the bounding box is the same as the center of the object.
        self.__Bounding_Box['Centroid'] = p
        
        return self.__Bounding_Box
    
    def __Update(self) -> None:
        """
        Description:
            Update the scene.
        """

        bpy.context.view_layer.update()
    
    def Reset(self) -> None:
        # ...
        self.__T = Transformation.Homogeneous_Transformation_Matrix_Cls(self.__Obj_Param_Str.T.all().copy(), np.float32)

        # ...
        Lib.Blender.Utilities.Set_Object_Transformation(self.__Obj_Param_Str.Name, self.__Obj_Param_Str.T)
        self.__Update()

    def Visibility(self, state: bool) -> None:
        Lib.Blender.Utilities.Object_Visibility(self.__Obj_Param_Str.Name, state)
        self.__Update()

    def Random(self) -> None:
        # ...
        p = np.zeros(3, np.float32); theta = p.copy()
        for i, (l_p_i, l_theta_i) in enumerate(zip(self.__Obj_Param_Str.Limit.Position.items(), 
                                                   self.__Obj_Param_Str.Limit.Rotation.items())):
            # ...
            if l_p_i[1] != None:
                p[i] = np.random.uniform(l_p_i[1][0], l_p_i[1][1])
            # ...
            if l_theta_i[1] != None:
                theta[i] = np.random.uniform(l_theta_i[1][0], l_theta_i[1][1])

        # ...
        self.__T = self.__Obj_Param_Str.T.Rotation(theta, self.__axes_sequence_cfg).Translation(p)

        # ...
        Lib.Blender.Utilities.Set_Object_Transformation(self.__Obj_Param_Str.Name, self.__T)
        self.__Update()



