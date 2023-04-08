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

for i, (l_p_i, l_theta_i) in enumerate(zip(Lib.Parameters.Object.T_Joint_VT_2_Str.Limit.Position.items(), 
                                           Lib.Parameters.Object.T_Joint_VT_2_Str.Limit.Rotation.items())):
    if l_p_i[1] != None:
        v = l_p_i[1]
        print(i, l_p_i[1][0])




