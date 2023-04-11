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
#   ../Lib/Blender/Utilities
import Lib.Blender.Utilities as Utilities
#   ../Lib/Transformation/Core
import Lib.Transformation.Core as Transformation
#   ../Lib/Parameters/Camera & Object
import Lib.Parameters.Camera
import Lib.Parameters.Object

# Focal length of spherical concave mirror is .. f = alpha / 2
# Focal length: f = 

# Warning ...
#   New Script Utilities -> add helper functions
#   New Class Bounding Box -> to work with boxes

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
        Utilities.Set_Object_Transformation(self.__name, self.__Cam_Param_Str.T)

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
    # https://blender.stackexchange.com/questions/16472/how-can-i-get-the-cameras-projection-matrix

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

    # https://ksimek.github.io/2012/08/22/extrinsic/
    # https://en.wikipedia.org/wiki/Camera_resectioning
    # https://miaodx.com/blogs/unrealcv_digest/camera_pose/
    # https://blender.stackexchange.com/questions/882/how-to-find-image-coordinates-of-the-rendered-vertex
    # https://github.com/vsitzmann/shapenet_renderer/blob/master/util.py
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

class Bounding_Box_Cls(object):
    def __init__(self):
        # Shape (Bounding Box):
        #   Vertices: 8; Space: 3D;
        self.__CONST_BOX_SHAPE = (8, 3)

class Object_Cls(object):
    """
    Description:
        ...
    """
    def __init__(self, Obj_Param_Str: Lib.Parameters.Object.Object_Parameters_Str, axes_sequence_cfg: str) -> None:
        self.__Obj_Param_Str = Obj_Param_Str
        self.__T = Transformation.Homogeneous_Transformation_Matrix_Cls(None, np.float32)
        self.__axes_sequence_cfg = axes_sequence_cfg

        # test something
        Utilities.Set_Object_Transformation(self.__Obj_Param_Str.Name, Transformation.Homogeneous_Transformation_Matrix_Cls(None, np.float32))

        # Update the scene.
        bpy.context.view_layer.update()

        # Calculate the bounding box parameters from the vertices of the main object.
        #   Get the minimum and maximum X, Y, Z values of the input vertices.
        (min_verts, max_verts) = Utilities.Get_Min_Max(np.array(Utilities.Get_Vertices_From_Object(self.__Obj_Param_Str.Name), dtype=np.float32))
        #   Properties of the object.
        self.__origin = np.array([(max_v_i + min_v_i)/2.0 for _, (min_v_i, max_v_i) in enumerate(zip(min_verts, max_verts))], dtype=np.float32)

        # ...
        self.__Bounding_Box_Size = np.array([np.abs(max_v_i - min_v_i) for _, (min_v_i, max_v_i) in enumerate(zip(min_verts, max_verts))], dtype=np.float32)

        # Convert the initial object sizes to a transformation matrix.
        self.__T_Size = Transformation.Homogeneous_Transformation_Matrix_Cls(None, np.float32).Scale(self.__Bounding_Box_Size)

        # Calculate the vertices of the box defined by the input parameters of the class.
        self.__Init_Bounding_Box_Vertices = np.zeros((8, 3), dtype=np.float32)
        for i, verts_i in enumerate(self.__Get_Init_Vertices()):
            self.__Init_Bounding_Box_Vertices[i, :] = (self.__T_Size.all() @ np.append(verts_i, 1.0).tolist())[0:3] + self.__origin

        self.__Bounding_Box_Vertices = np.zeros((8, 3), dtype=np.float32)

        # ....
        self.Reset()

        # ...
        self.__Transformation_Bounding_Box()

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
        The main parametrs of the bounding box.
        """

        return {'Centroid': self.__T.p.all(), 'Size': self.__Bounding_Box_Size, 'Vertices': self.__Bounding_Box_Vertices}
    
    @staticmethod
    def __Get_Init_Vertices() -> tp.List[tp.List[float]]:
        """
        Description:
            A helper function to get the initial vertices of the bounding box of an object.

            Note: 
                Lower Base: A {id: 0}, B {id: 1}, C {id: 2}, D {id: 3}
                Upper Base: E {id: 4}, F {id: 5}, G {id: 6}, H {id: 7}

        Returns:
            (1) parameter [Vector<float> 8x3]: Vertices of the object.
        """
 
        return np.array([[0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5],
                         [0.5, -0.5,  0.5], [0.5, 0.5,  0.5], [-0.5, 0.5,  0.5], [-0.5, -0.5,  0.5]], dtype=np.float32)
    
    def __Transformation_Bounding_Box(self) -> tp.List[tp.List[float]]:
        """
        Description:
            ...
        """

        q = self.__T.Get_Rotation('QUATERNION'); p = self.__T.p.all().copy()
        for i, verts_i in enumerate(self.__Init_Bounding_Box_Vertices):
            self.__Bounding_Box_Vertices[i, :] = q.Rotate(Transformation.Vector3_Cls(verts_i, np.float32)).all() + p

    def Reset(self) -> None:
        self.__T = Transformation.Homogeneous_Transformation_Matrix_Cls(self.__Obj_Param_Str.T.all().copy(), np.float32)

        Utilities.Set_Object_Transformation(self.__Obj_Param_Str.Name, self.__Obj_Param_Str.T)

        # Update the scene.
        bpy.context.view_layer.update()

    # add -> visibility of the bounding box
    # change the name!!! Vis_BB is just for test
    def Vis_BB(self):
        # Properties of the created object.
        box_properties = {'transformation': {'Size': 1.0, 'Scale': self.__Bounding_Box_Size, 'Location': [0.0, 0.0, 0.0]}, 
                          'material': {'RGBA': [0.8,0.8,0.8,1.0], 'alpha': 1.0}}
            
        Utilities.Create_Primitive('Cube', f'{self.__Obj_Param_Str.Name}_Bounding_Box', box_properties)

        bpy.data.objects[f'{self.__Obj_Param_Str.Name}_Bounding_Box'].rotation_mode = 'ZYX'

        # It works!
        Utilities.Set_Object_Origin(f'{self.__Obj_Param_Str.Name}_Bounding_Box', (-1) * self.__origin)

        # ...
        Utilities.Set_Object_Transformation(f'{self.__Obj_Param_Str.Name}_Bounding_Box', self.__T)

    def Visibility(self, state: bool) -> None:
        Utilities.Object_Visibility(self.__Obj_Param_Str.Name, state)
        
    def Random(self) -> None:
        p = np.zeros(3, np.float32); theta = p.copy()
        for i, (l_p_i, l_theta_i) in enumerate(zip(self.__Obj_Param_Str.Limit.Position.items(), 
                                                   self.__Obj_Param_Str.Limit.Rotation.items())):
            if l_p_i[1] != None:
                p[i] = np.random.uniform(l_p_i[1][0], l_p_i[1][1])

        
            if l_theta_i[1] != None:
                theta[i] = np.random.uniform(l_theta_i[1][0], l_theta_i[1][1])

        self.__T = self.__Obj_Param_Str.T.Rotation(theta, self.__axes_sequence_cfg).Translation(p)

        Utilities.Set_Object_Transformation(self.__Obj_Param_Str.Name, self.__T)

        self.__Transformation_Bounding_Box()

        # Update the scene.
        bpy.context.view_layer.update()



