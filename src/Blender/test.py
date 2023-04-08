# Custom script to test some functions, classes, etc.

# System (Default)
import sys
sys.path.append('..')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# ...
import Lib.Camera.Parameters

print(Lib.Camera.Parameters.PhoXi_Scanner_M_Str.Resolution['x'])