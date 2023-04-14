# BPY (Blender as a python) [pip3 install bpy]
import bpy
# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../' + 'src')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Custom Library:
#   ../Lib/Blender/Core
import Lib.Blender.Core
#   ../Lib/Camera/Parameters
import Lib.Camera.Parameters
#   ../Lib/Transformation/Core
import Lib.Transformation.Core as Transformation
#   ../Lib/Transformation/Utilities/Mathematics
import Lib.Transformation.Utilities.Mathematics as Mathematics

def main():
    pass

if __name__ == '__main__':
    main()