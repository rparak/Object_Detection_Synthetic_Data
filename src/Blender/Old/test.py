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
# ...
import Lib.Utilities.General as General


Bounding_Box_YOLO = {'x_c': 0.504347, 'y_c': 0.442419, 'width': 0.087728, 'height': 0.089276}
Bounding_Box_PASCAL_VOC = {'x_min': 950, 'y_min': 614, 'x_max': 1132, 'y_max': 752}
Resolution = {'x': 2064, 'y': 1544}

res = General.Convert_Annotation('YOLO', 'PASCAL_VOC', Bounding_Box_YOLO, Resolution)
print(res)




