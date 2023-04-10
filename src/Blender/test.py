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

# p = [-0.08269086, -0.03376902, 0.] 
# theta =[ 0., 0., -2.4014683]
T_0 = Transformation.Homogeneous_Transformation_Matrix_Cls([[ 0.99939,  0.,       0.0349,   0.175  ],
                                                            [ 0.,       1.,       0.,       0.     ],
                                                            [-0.0349 ,  0.,       0.99939,  0.0193 ],
                                                            [ 0.,       0.,       0.,       1.     ]], np.float32)

T_1 = T_0.Rotation([ 0., 0., -2.4014683], 'ZYX').Translation([-0.08269086, -0.03376902, 0.])

print(T_1)
print(T_0)

"""
print(T_0 @ T_1)
print('Position')
print(T_0.p, T_1.p)
print((T_0 @ T_1).p)
print('Rotation')
print(T_0.Get_Rotation('ZYX').Degree)
print(T_1.Get_Rotation('ZYX').Degree)
print((T_0 @ T_1).Get_Rotation('ZYX').Degree)
"""





