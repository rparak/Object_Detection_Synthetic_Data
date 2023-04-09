# Custom script to test some functions, classes, etc.

# System (Default)
import sys
sys.path.append('..')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# ...
import Lib.Parameters.Camera
# ...
import Lib.Parameters.Object
# ...
import Lib.Transformation.Core as Transformation

print(Lib.Parameters.Camera.PhoXi_Scanner_M_Str.Resolution['x'])

print(Lib.Parameters.Object.T_Joint_VT_2_Str.T.Get_Rotation('ZYX').Degree)

print(Lib.Parameters.Object.T_Joint_VT_2_Str.Limit.Position)





