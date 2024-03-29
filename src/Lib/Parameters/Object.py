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
File Name: Object.py
## =========================================================================== ## 
"""

# Numpy (Array computing) [pip3 install numpy]tp
import numpy as np
# Dataclasses (Data Classes)
from dataclasses import dataclass, field
# Typing (Support for type hints)
import typing as tp
# Custom Library:
#   ../Lib/Transformation/Core
import Lib.Transformation.Core as Transformation 
#   ../Lib/Transformation/Utilities
import Lib.Transformation.Utilities.Mathematics as Mathematics
#   ../Lib/Parameters/Primitives
import Lib.Parameters.Primitives as Primitives

@dataclass
class Limit_Str:
    """
    Description:
        An object structure that defines the boundaries of an object's position/rotation.

        Note:
            The individual parts of the structure shall have the following shape:
                {'x': [x_{-}, x_{+}], 'y': [y_{-}, y_{+}], 'z': [z_{-}, z_{+}}
            
            If the object does not have a boundary in the axis, just mark it as None. For example, if there 
            is no boundary in the 'y' axis, then:
                {'x': [x_{-}, x_{+}], 'y': None, 'z': [z_{-}, z_{+}}
    """
        
    Position: tp.Tuple[tp.List[float], tp.List[float], 
                       tp.List[float]] = field(default_factory=tuple)
    Rotation: tp.Tuple[tp.List[float], tp.List[float], 
                       tp.List[float]] = field(default_factory=tuple)

@dataclass
class Object_Parameters_Str:
    """
    Description:
        The structure of the main parameters of the scanned object.
    """

    # The name of the object in Blender.
    Name: str = ''
    # The identification number (ID) of the object.
    Id: int = 0
    # Homogeneous transformation matrix of the initial position of the object.
    T: tp.List[tp.List[float]] = field(default_factory=list)
    # Parameters of the object bounding box. Generated from the script gen_object_bounding_box.py. 
    Bounding_Box: Primitives.Box_Cls = field(default_factory=Primitives.Box_Cls)
    # The position/rotation boundaries of the object.
    Limit: Limit_Str = field(default_factory=Limit_Str)

"""
Description:
    The main parameters of the object: T-Joint
"""
T_Joint_001_Str = Object_Parameters_Str(Name='T_Joint_001', Id=0)
# Homogeneous transformation matrix {T} of the object.
T_Joint_001_Str.T = Transformation.Homogeneous_Transformation_Matrix_Cls(None, np.float64).Rotation([0.0, Mathematics.Degree_To_Radian(2.0), 0.0], 
                                                                                                    'ZYX').Translation([0.175, 0.0, 0.0193])
# Parameters of the object bounding box.
T_Joint_001_Str.Bounding_Box = Primitives.Box_Cls([-0.00545, 0.0, 0.0], [0.0475, 0.0584, 0.0366])
# Other parameters.
T_Joint_001_Str.Limit.Position = {'x': [-0.2, 0.2], 'y': [-0.325, 0.4], 'z': None}
T_Joint_001_Str.Limit.Rotation = {'x': None, 'y': None, 'z': [-Mathematics.CONST_MATH_PI, Mathematics.CONST_MATH_PI]}

"""
Description:
    The main parameters of the object: Metan Blank
"""
Metal_Blank_001_Str = Object_Parameters_Str(Name='Metal_Blank_001', Id=1)
# Homogeneous transformation matrix {T} of the object.
Metal_Blank_001_Str.T = Transformation.Homogeneous_Transformation_Matrix_Cls(None, np.float64).Rotation([0.0, 0.0, 0.0], 
                                                                                                        'ZYX').Translation([0.175, 0.0, 0.001])
# Parameters of the object bounding box.
Metal_Blank_001_Str.Bounding_Box = Primitives.Box_Cls([0.0, 0.0, -0.007], [0.05, 0.05, 0.014])
# Other parameters.
Metal_Blank_001_Str.Limit.Position = {'x': [-0.2, 0.2], 'y': [-0.325, 0.4], 'z': None}
Metal_Blank_001_Str.Limit.Rotation = {'x': None, 'y': None, 'z': [-Mathematics.CONST_MATH_PI, Mathematics.CONST_MATH_PI]}
