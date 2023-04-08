# Custom script to test some functions, classes, etc.

# System (Default)
import sys
sys.path.append('..')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# ...
import Lib.Camera.Parameters
# ...
import Lib.Transformation.Core as Transformation

print(Lib.Camera.Parameters.PhoXi_Scanner_M_Str.Resolution['x'])

T_0 = Transformation.Homogeneous_Transformation_Matrix_Cls(None, np.float32)
T = T_0.Translation(Lib.Camera.Parameters.PhoXi_Scanner_M_Str.Location.all()).Rotation(Lib.Camera.Parameters.PhoXi_Scanner_M_Str.Rotation.all(),'ZYX')


print(T.Get_Rotation('ZYX').Degree)