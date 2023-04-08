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

    # Transformation parameters of the object. Cartesian coordinates.
    #   Position - p(x, y, z) in meters.
    Position: Transformation.Vector3_Cls = field(default_factory=list)
    #   Rotation (Euler Angles) - theta(z, y, x) in radians.
    Rotation: Transformation.Euler_Angle_Cls = field(default_factory=list)
    # Camera resolution in pixels (x, y).
    Resolution: tp.Dict[int, int] = field(default_factory=dict)
    #  The properties of the object.
    #   Projection of the camera's field of view: Perspective = ['PERSP'], Orthographic = ['ORTHO']
    Type: str = 'PERSP'
    #   An exponential brightness factor.
    Gamma: float = 1.0
    #   Camera projection angle in radians.
    alpha: float = 0.0
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
# Cartesian coordinates.
#   Location in meters.
PhoXi_Scanner_M_Str.Position = Transformation.Vector3_Cls([0.145, -0.175, 0.885], np.float32)
#   Rotation in radians.
PhoXi_Scanner_M_Str.Rotation = Transformation.Euler_Angle_Cls([0.0, Mathematics.Degree_To_Radian(11.75), -Mathematics.CONST_MATH_HALF_PI], 
                                                              'ZYX', np.float32)
# Camera resolution in pixels.
PhoXi_Scanner_M_Str.Resolution = {'x': 2064, 'y': 1544}
# The properties of the camera:
PhoXi_Scanner_M_Str.Type = 'PERSP'
PhoXi_Scanner_M_Str.Gamma = 1.0
PhoXi_Scanner_M_Str.alpha = Mathematics.Degree_To_Radian(78.25)
PhoXi_Scanner_M_Str.Spectrum = 'Monochrome'