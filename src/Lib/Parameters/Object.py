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

"""
[INFO] Object name: Metal_Blank_001
[INFO] Bounding Box Parameters:
[INFO]  - Origin: [0.    0.    0.007]
[INFO]  - Size: [0.05  0.05  0.014]
[INFO] The bounding box was successfully created!
[INFO] Object name: T_Joint_001
[INFO] Bounding Box Parameters:
[INFO]  - Origin: [0.00545 0.      0.     ]
[INFO]  - Size: [0.0475 0.0584 0.0366]
[INFO] The bounding box was successfully created!
"""

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
    # Identification number.
    Id: int = 0
    # Homogeneous transformation matrix of the initial position of the object.
    T: tp.List[tp.List[float]] = field(default_factory=list)
    # The position/rotation boundaries of the object.
    Limit: Limit_Str = field(default_factory=Limit_Str)

"""
Description:
    The main parameters of the T-Joint object in view type 0.
"""
T_Joint_VT_0_Str = Object_Parameters_Str(Name='T_Joint_VT_0', Id=0)
# Homogeneous transformation matrix {T} of the object.
T_Joint_VT_0_Str.T = Transformation.Homogeneous_Transformation_Matrix_Cls(None, np.float32).Rotation([0.0, Mathematics.Degree_To_Radian(2.0), 0.0], 
                                                                                                     'ZYX').Translation([0.175, 0.0, 0.0193])
# Other parameters.
T_Joint_VT_0_Str.Limit.Position = {'x': [-0.2, 0.2], 'y': [-0.325, 0.4], 'z': None}
T_Joint_VT_0_Str.Limit.Rotation = {'x': None, 'y': None, 'z': [-Mathematics.CONST_MATH_PI, Mathematics.CONST_MATH_PI]}

"""
Description:
    The main parameters of the T-Joint object in view type 1.
"""
T_Joint_VT_1_Str = Object_Parameters_Str(Name='T_Joint_VT_1', Id=1)
# Homogeneous transformation matrix {T} of the object.
T_Joint_VT_1_Str.T = Transformation.Homogeneous_Transformation_Matrix_Cls(None, np.float32).Rotation([Mathematics.CONST_MATH_HALF_PI, 0.0, 0.0], 
                                                                                                     'ZYX').Translation([0.175, 0.0, 0.0302])
# Other parameters.
T_Joint_VT_1_Str.Limit.Position = {'x': [-0.2, 0.2], 'y': [-0.325, 0.4], 'z': None}
T_Joint_VT_1_Str.Limit.Rotation = {'x': None, 'y': [-Mathematics.CONST_MATH_PI, Mathematics.CONST_MATH_PI], 'z': None}

"""
Description:
    The main parameters of the T-Joint object in view type 2.
"""
T_Joint_VT_2_Str = Object_Parameters_Str(Name='T_Joint_VT_2', Id=2)
# Homogeneous transformation matrix {T} of the object.
T_Joint_VT_2_Str.T = Transformation.Homogeneous_Transformation_Matrix_Cls(None, np.float32).Rotation([Mathematics.CONST_MATH_HALF_PI, 0.0, -Mathematics.CONST_MATH_HALF_PI], 
                                                                                                     'ZYX').Translation([0.175, 0.0, 0.0302])
# Other parameters.
T_Joint_VT_2_Str.Limit.Position = {'x': [-0.2, 0.2], 'y': [-0.325, 0.4], 'z': None}
T_Joint_VT_2_Str.Limit.Rotation = {'x': None, 'y': [-Mathematics.CONST_MATH_PI, Mathematics.CONST_MATH_PI], 'z': None}
