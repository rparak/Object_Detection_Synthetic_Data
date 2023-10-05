"""
## =========================================================================== ## 
MIT License
Copyright (c) 2023 Roman Parak
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
## =========================================================================== ## 
Author   : Roman Parak
Email    : Roman.Parak@outlook.com
Github   : https://github.com/rparak
File Name: Core.py
## =========================================================================== ## 
"""

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
        A class for working with a camera object in a Blender scene.

        The main part is to solve the equation:

            P = K x [R | t],
        
        where {K} is the instrict matrix that contains the intrinsic parameters, and {[R | t]} is the extrinsic 
        matrix that is combination of rotation matrix {R} and a translation vector {t}.

    Initialization of the Class:
        Args:
            (1) name [string]: Object name.
            (2) Cam_Param_Str [Camera_Parameters_Str]: The structure of the main parameters of the camera (sensor) object.
                                                       Note:
                                                        See the ../Lib/Parameters/Camera script.
            (3) image_format [string]: The format to save the rendered image.

        Example:
            Initialization:
                # Assignment of the variables.
                name = 'Camera'
                Camera_Str = Camera_Parameters_Str()
                image_format = 'PNG'

                # Initialization of the class.
                Cls = Camera_Cls(name, Camera_Str, image_format)

            Features:
                # Properties of the class.
                None

                # Functions of the class.
                Cls.K; Cls.R_t
                    ...
                Cls.P
    """
        
    def __init__(self, name: str, Cam_Param_Str: Lib.Parameters.Camera.Camera_Parameters_Str, image_format: str = 'PNG') -> None:
        try:
            assert Lib.Blender.Utilities.Object_Exist(name) == True
            
            # << PRIVATE >> #
            # The name of the camera object (in Blender).
            self.__name = name
            # The structure of the main parameters of the camera (sensor) object.
            self.__Cam_Param_Str = Cam_Param_Str
            # The format to save the rendered image.
            self.__image_format = image_format

            # Set the main parameters of the camera object.
            self.__Set_Camera_Parameters()

        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
            print(f'[ERROR] The camera object named <{name}> does not exist in the current scene.')

    def __Update(self) -> None:
        """
        Description:
            Update the scene.
        """

        bpy.context.view_layer.update()

    def __Set_Camera_Parameters(self) -> None:
        """
        Description:
            Set the main parameters of the camera object.
        """

        # Set the transformation of the object.
        Lib.Blender.Utilities.Set_Object_Transformation(self.__name, self.__Cam_Param_Str.T)
        self.__Update()

        # Adjust the width or height of the sensor depending on the resolution of the image.
        bpy.data.cameras[self.__name].sensor_fit = 'AUTO'
        # Resolution of the image:
        #   ['x']: Horizontal pixels.
        bpy.context.scene.render.resolution_x = self.__Cam_Param_Str.Resolution['x']
        #   ['y']: Vertical pixels.
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
        # The format to save the rendered image.
        bpy.context.scene.render.image_settings.file_format = self.__image_format
        # Set the type of device to be used to render the image.
        bpy.context.scene.cycles.device = 'GPU'
        # Set additional rendering parameters.
        #   Note:
        #       Non-ideal parameters of the rendering process can be useful in training.
        if bpy.context.scene.cycles.use_adaptive_sampling == False:
            bpy.context.scene.cycles.use_adaptive_sampling = True
        # Ideal parameters:
        #   adaptive_threshold = 0.001
        #   samples = 512
        # Non-Ideal parameters (images with additional noise):
        #   adaptive_threshold = 0.025
        #   samples = 128
        bpy.context.scene.cycles.adaptive_threshold = 0.025
        bpy.context.scene.cycles.samples = 128
        
        #  Update the scene.
        self.__Update()

    def K(self) -> tp.List[tp.List[float]]:
        """
        Description:
            Get the intrinsic matrix {K}, which contains the intrinsic parameters.

            Equation:
                K = [[alpha_u,   gamma, u_0],
                     [      0, alpha_v, v_0],
                     [      0,       0,   1]],

                where (alpha_u, alpha_v) are focal lengths expressed in units of pixels (note: usually the same), (u_0, v_0) are the principal 
                point (i.e. the central point of the image frame) and gamma is the skew between the axes (note: usually equal to zero).

        Returns:
            (1) parameter [Matrix<float> 3x3]: Instrict matrix of the camera.
        """

        try:
            assert bpy.data.cameras[self.__name].sensor_fit == 'AUTO'

            # Express the parameters of intrinsic matrix {K} of the camera.
            gamma = 0.0
            #   Focal lengths expressed in units of pixesl: alpha_u and alpha_v (note: alpha_v = alpha_u)
            alpha_u = (self.__Cam_Param_Str.f * self.__Cam_Param_Str.Resolution['x']) / bpy.data.cameras[self.__name].sensor_width
            alpha_v = alpha_u
            #   Principal point: u_0 and v_0
            u_0 = self.__Cam_Param_Str.Resolution['x'] / 2.0
            v_0 = self.__Cam_Param_Str.Resolution['y'] / 2.0

            return Transformation.Homogeneous_Transformation_Matrix_Cls([[alpha_u,   gamma, u_0, 0.0],
                                                                         [    0.0, alpha_v, v_0, 0.0],
                                                                         [    0.0,     0.0, 1.0, 0.0],
                                                                         [    0.0,     0.0, 0.0, 1.0]], np.float64).R
        
        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
            print('[ERROR] Incorrectly set method to fit the image and field of view angle inside the sensor.')
            print(f'[ERROR] The method must be set to AUTO. Not to {bpy.data.cameras[self.__name].sensor_fit}')

    def R_t(self) -> tp.List[tp.List[float]]:
        """
        Description:
            Get the extrinsic matrix {[R | t]}, which is the combination of the rotation matrix {R} and a translation 
            vector {t}.

            The standard form of the homogeneous transformation matrix {T}:
                T = [R_{3x3}, t_{3x1}
                     0_{1x3}, 1_{1x1}],

                where R is a rotation matrix and t is a translation vector.

            The inverse form of the homogeneous transformation matrix:
                T^(-1) = [R^T_{3x3}, -R^T_{3x3} x t_{3x1}
                            0_{1x3},              1_{1x1}]

            The relationship between the extrinsic matrix parameters and the position 
            of the camera is:
                [R | t] = [R_{C} | C]^(-1),
                
                where C is a column vector describing the position of the camera center in world coordinates 
                and R_{C} is a rotation matrix describing the camera orientation.

            then we can express the parameters R, t as:
                R = R_{C}^T
                t = -R_{C}^T x C
            
        Returns:
            (1) parameter [Matrix<float> 3x4]: Extrinsic matrix of the camera.
        """
                
        # Modification matrix {R} to adjust the direction (sign {+, -}) of each axis.  
        R_mod = np.array([[1.0,  0.0,  0.0],
                          [0.0, -1.0,  0.0],
                          [0.0,  0.0, -1.0]], dtype=np.float64)

        # Expression of the parameters R, t of the extrinsic matrix.
        R = R_mod @ self.__Cam_Param_Str.T.Transpose().R
        t = (-1) * R @ self.__Cam_Param_Str.T.p.all()

        return np.hstack((R, t.reshape(3, 1)))

    def P(self) -> tp.List[tp.List[float]]:
        """
        Description:
            Get the projection matrix {P} of the camera object.

            Equation:
                P = K x [R | t]

        Returns:
            (1) parameter [Matrix<float> 3x4]: Projection matrix of the camera.
        """
                
        return self.K() @ self.R_t()
    
    def Random(self) -> None:
        """
        Description:
            A function to randomly generate camera properties as well as to generate illumination.

            Note:
                The function can be easily modified/extended with additional functions.
        """

        # The strength of the light (the material of the object to be used for illumination).
        bpy.data.materials['Light'].node_tree.nodes['Emission'].inputs[1].default_value = np.float64(np.random.uniform(7.5 - 1.0, 
                                                                                                                       7.5 + 1.0))

        # Non-Ideal parameters (images with additional noise):
        #   adaptive_threshold = 0.02 .. 0.03
        bpy.context.scene.cycles.adaptive_threshold = np.float64(np.random.uniform(0.025 - 0.005, 
                                                                                   0.025 + 0.005))

        #  Update the scene.
        self.__Update()

class Object_Cls(object):
    """
    Description:
        A class for working with a scanned object in a Blender scene.

    Initialization of the Class:
        Args:
            (1) Obj_Param_Str [Object_Parameters_Str]: The structure of the main parameters of the scanned object.
                                                       Note:
                                                        See the ../Lib/Parameters/Object script.
            (2) axes_sequence_cfg [string]: Rotation axis sequence configuration (e.g. 'ZYX', 'QUATERNION', etc.)

        Example:
            Initialization:
                # Assignment of the variables.
                Object_Str = Object_Parameters_Str()
                axes_sequence_cfg = 'ZYX'

                # Initialization of the class.
                Cls = Object_Cls(Object_Str, axes_sequence_cfg)

            Features:
                # Properties of the class.
                self.T; self.T_0
                    ...
                self.Bounding_Box

                # Functions of the class.
                Cls.Reset(); Cls.Visibility(True)
                    ...
                Cls.Random()
    """
    
    def __init__(self, Obj_Param_Str: Lib.Parameters.Object.Object_Parameters_Str, axes_sequence_cfg: str) -> None:
        try:
            assert Lib.Blender.Utilities.Object_Exist(Obj_Param_Str.Name) == True

            # << PRIVATE >> #
            self.__axes_sequence_cfg = axes_sequence_cfg
            # The structure of the main parameters of the scanned object.
            self.__Obj_Param_Str = Obj_Param_Str

            # Initialize the homogeneous transformation matrix as the identity matrix.
            self.__T = Transformation.Homogeneous_Transformation_Matrix_Cls(None, np.float64)

            # Create a dictionary and initialize the object's bounding box parameters.
            self.__Bounding_Box = {'Centroid': self.__T.p.all(), 'Size': self.__Obj_Param_Str.Bounding_Box.Size, 
                                   'Vertices': self.__Obj_Param_Str.Bounding_Box.Vertices.copy()}

            # Set the object transform to zero position.
            Lib.Blender.Utilities.Set_Object_Transformation(self.__Obj_Param_Str.Name, Transformation.Homogeneous_Transformation_Matrix_Cls(None, np.float64))
            self.__Update()

            # Return the object to the initialization position.
            self.Reset()
        
        except AssertionError as error:
            print(f'[ERROR] Information: {error}')
            print(f'[ERROR] An object named <{Obj_Param_Str.Name}> does not exist in the current scene.')

    @property
    def Name(self) -> str:
        """
        Description:
            Get the name of the object.

        Returns:
            (1) parameter [string]: Name of the object.
        """

        return self.__Obj_Param_Str.Name

    @property
    def Id(self) -> int:
        """
        Description:
            Get the identification number of the object.

        Returns:
            (1) parameter [int]: Identification number.
        """

        return self.__Obj_Param_Str.Id
    
    @property
    def T_0(self) -> tp.List[tp.List[float]]:
        """
        Description:
            Get the initial (null) homogeneous transformation matrix of an object.

        Returns:
            (1) parameter [Matrix<float> 4x4]: Homogeneous transformation matrix.
        """

        return self.__Obj_Param_Str.T
    
    @property
    def T(self):
        """
        Description:
            Get the actual homogeneous transformation matrix of an object.

        Returns:
            (1) parameter [Matrix<float> 4x4]: Homogeneous transformation matrix.
        """
                
        return self.__T
    
    @property
    def Vertices(self):
        """
        Description:
            Get the positions (x, y, z) of the vertices of the given object.

        Returns:
            (1) parameter [Matrix<float> 3xn]: The vertices of the scanned object.
                                               Note:
                                                Where n is the number of vertices.
        """
                
        return np.array(Lib.Blender.Utilities.Get_Vertices_From_Object(self.__Obj_Param_Str.Name),
                        dtype=np.float64)
    
    @property
    def Bounding_Box(self) -> tp.Tuple[tp.List[float], tp.List[float], tp.List[tp.List[float]]]:
        """
        Description:
            Get the main parameters of the object's bounding box.

        Returns:
            (1) parameter [Dictionary {'Centroid': Vector<float> 1x3, 'Size': Vector<float> 1x3, 
                                       'Vertices': Matrix<float> 3x8}]: The main parameters of the bounding box as a dictionary.
        """

        # Oriented Bounding Box (OBB) transformation according to the homogeneous transformation matrix of the object.
        q = self.__T.Get_Rotation('QUATERNION'); p = self.__T.p.all()
        for i, point_i in enumerate(self.__Obj_Param_Str.Bounding_Box.Vertices):
            self.__Bounding_Box['Vertices'][i, :] = q.Rotate(Transformation.Vector3_Cls(point_i, np.float64)).all() + p

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
        """
        Description:
            Function to return the object to the initialization position.
        """

        self.__T = Transformation.Homogeneous_Transformation_Matrix_Cls(self.__Obj_Param_Str.T.all().copy(), np.float64)

        # Set the transformation of the object to the initial position.
        Lib.Blender.Utilities.Set_Object_Transformation(self.__Obj_Param_Str.Name, self.__Obj_Param_Str.T)
        self.__Update()

    def Visibility(self, state: bool) -> None:
        """
        Description:
            Function to enable and disable the visibility of an object.

        Args:
            (1) state [bool]: Enable (True) / Disable (False). 
        """

        Lib.Blender.Utilities.Object_Visibility(self.__Obj_Param_Str.Name, state)
        self.__Update()

    def Random(self) -> None:
        """
        Description:
            Function for random generation of object transformation (position, rotation). The boundaries of the random 
            generation are defined in the object structure.

            Note:
                If there are no boundaries in any axis (equals None), continue without generating.
        """

        p = np.zeros(3, np.float64); theta = p.copy()
        for i, (l_p_i, l_theta_i) in enumerate(zip(self.__Obj_Param_Str.Limit.Position.items(), 
                                                   self.__Obj_Param_Str.Limit.Rotation.items())):
            # Random Position: {p}
            if l_p_i[1] != None:
                p[i] = np.random.uniform(l_p_i[1][0], l_p_i[1][1])

            # Random Rotation: {theta}
            if l_theta_i[1] != None:
                theta[i] = np.random.uniform(l_theta_i[1][0], l_theta_i[1][1])

        # Create a homogeneous transformation matrix from random values.
        self.__T = self.__Obj_Param_Str.T.Rotation(theta, self.__axes_sequence_cfg).Translation(p)

        # Set the object transformation.
        Lib.Blender.Utilities.Set_Object_Transformation(self.__Obj_Param_Str.Name, self.__T)
        self.__Update()



