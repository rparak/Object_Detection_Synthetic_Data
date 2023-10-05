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
File Name: Camera.py
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

@dataclass
class Camera_Parameters_Str:
    """
    Description:
        The structure of the main parameters of the camera (sensor) object.

        Note:
            The parameter structure mainly focuses on the 2D representation of the image.
    """

    # Homogeneous transformation matrix of the object.
    T: tp.List[tp.List[float]] = field(default_factory=list)
    # Camera resolution in pixels (x, y).
    Resolution: tp.Dict[int, int] = field(default_factory=dict)
    #  The properties of the object.
    #   Projection of the camera's field of view: Perspective = ['PERSP'], Orthographic = ['ORTHO']
    Type: str = 'PERSP'
    #   An exponential brightness factor.
    Gamma: float = 1.0
    #   Camera projection angle in degrees.
    alpha: float = 0.0
    #   The focal length (lens) of the camera in millimeters.
    f: float = 0.0
    #   Spectrum of the camera (Monochrome / Color). The color parameter of the output image.
    Spectrum: str = 'Monochrome'

"""
Description:
    Parameters of the Photoneo PhoXi 3D Scanner M.

    More information about the scanner can be found on the official website here:
        https://www.photoneo.com/

    Available parameters for the PhoXi 3D Scanner in the individual type:
        Resolution: 
            Low{'x': 1032px, 'y': 772px}, High{'x': 2064px, 'y': 1544px}
        Camera mounting angle:
            {'XS': 0.0° (15.45°), 'S': 15.45°, 'M': 11.75°, 'L': 9.45°, 'XL': 7.50°}
        Projection angle:
            {'XS': 73°, 'S': 74.55°, 'M': 78.25°, 'L': 80.55°, 'XL': 82.5°}
"""
PhoXi_Scanner_M_Str = Camera_Parameters_Str()
# Homogeneous transformation matrix {T} of the object.
PhoXi_Scanner_M_Str.T = Transformation.Homogeneous_Transformation_Matrix_Cls(None, np.float64).Rotation([0.0, Mathematics.Degree_To_Radian(11.75), -Mathematics.CONST_MATH_HALF_PI], 
                                                                                                        'XYZ').Translation([0.145, -0.175, 0.885])
# Camera resolution in pixels.
PhoXi_Scanner_M_Str.Resolution = {'x': 2064, 'y': 1544}
# The properties of the camera:
PhoXi_Scanner_M_Str.Type = 'PERSP'
PhoXi_Scanner_M_Str.Gamma = 1.0
PhoXi_Scanner_M_Str.alpha = 78.25
PhoXi_Scanner_M_Str.f = PhoXi_Scanner_M_Str.alpha / 2.0
PhoXi_Scanner_M_Str.Spectrum = 'Monochrome'