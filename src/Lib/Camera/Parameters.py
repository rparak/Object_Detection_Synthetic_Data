# Numpy (Array computing) [pip3 install numpy]tp
import numpy as np
# Dataclasses (Data Classes)
from dataclasses import dataclass, field
# Typing (Support for type hints)
import typing as tp
# Custom Library:
#   ../Lib/Utilities/Transformation
import Lib.Utilities.Transformation as Transformation 


"""
Description:
    ....

    Something like a .... Parameters of the 2D camera, etc.
"""

@dataclass
class Camera_Parameters_Str:
    pass


# https://wiki.photoneo.com/index.php/PhoXi_3D_scanners_family

PhoXi_Scanner_M_Str = Camera_Parameters_Str()

# Resolution:
#   0: 1032x772
#   1: 2064x1544

# Gamma ...

# https://github.com/photoneo/phoxi_camera/blob/e8c403466cacce81c0fd24e167075ed2524837c2/urdf/PhoXi3Dscanner_values.xacro
# https://photoneo.com/files/manuals/Coordinate-spaces-Quick-Intro-PhoXi3DScanner.pdf
# https://www.photoneo.com/products/phoxi-scan-m/

# Projection angle: alpha = 78.25

# Focal length of spherical concave mirror is .. f = alpha / 2
# Focal length: f = 
