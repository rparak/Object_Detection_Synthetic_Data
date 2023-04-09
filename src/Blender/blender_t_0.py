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
#   ../Lib/Parameters/Camera & Object
import Lib.Parameters.Camera
import Lib.Parameters.Object
#   ../Lib/Transformation/Core
import Lib.Transformation.Core as Transformation
#   ../Lib/Transformation/Utilities/Mathematics
import Lib.Transformation.Utilities.Mathematics as Mathematics

def main():
    T_Joint_Id_0_Cls = Lib.Blender.Core.Object_Cls(Lib.Parameters.Object.T_Joint_VT_0_Str, 'ZYX')
    print(T_Joint_Id_0_Cls.T_0)
    T_Joint_Id_0_Cls.Reset()

if __name__ == '__main__':
    main()

