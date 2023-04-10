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

import bpy_extras
from mathutils import Matrix, Vector

def main():
    Camera_Cls = Lib.Blender.Core.Camera_Cls('Camera', Lib.Parameters.Camera.PhoXi_Scanner_M_Str, 'PNG')

    T_Joint_Id_0_Cls = Lib.Blender.Core.Object_Cls(Lib.Parameters.Object.T_Joint_VT_0_Str, 'ZYX')
    T_Joint_Id_0_Cls.Reset()
    

    """
    Lib.Blender.Core.Add_Primitive('Cube', {'Size': 1.0, 'Scale': T_Joint_Id_0_Cls.Bounding_Box['Size'], 
                                   'Location': T_Joint_Id_0_Cls.T.p.all()})
    """
    

    for i, verts_i in enumerate(T_Joint_Id_0_Cls.Bounding_Box['Vertices']):
        bpy.data.objects[f'Sphere_{i+1}'].location = verts_i
        #pass

    
if __name__ == '__main__':
    main()

