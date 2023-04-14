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
    scene = bpy.context.scene
    obj = bpy.data.objects['Camera']
    co = bpy.data.objects['T_Joint_VT_0'].location

    co_2d = bpy_extras.object_utils.world_to_camera_view(scene, obj, co)
    #print("2D Coords:", co_2d)
    
    # If you want pixel coords
    render_scale = scene.render.resolution_percentage / 100
    render_size = (
        int(scene.render.resolution_x * render_scale),
        int(scene.render.resolution_y * render_scale),
    )
    print("Pixel Coords:", (
          round(co_2d.x * render_size[0]),
          round(co_2d.y * render_size[1]),
    ))

    #T_Joint_Id_0_Cls = Lib.Blender.Core.Object_Cls(Lib.Parameters.Object.T_Joint_VT_0_Str, 'ZYX')
    #print(T_Joint_Id_0_Cls.T_0)
    #T_Joint_Id_0_Cls.Reset()
    
    Camera_Cls = Lib.Blender.Core.Camera_Cls('Camera', Lib.Parameters.Camera.PhoXi_Scanner_M_Str, 'PNG')
    P = Camera_Cls.P()

    #print(Matrix(P))
    #print(Matrix(Camera_Cls.K()))
    #print(Matrix(Camera_Cls.Rt()))
    
    p_1 = Matrix(P) @ co
    p_1 = np.vstack((P, np.ones(4))) @ np.hstack((np.array(co), 1))
    p_f = p_1.copy()
    print(Vector(p_1))
    p_1 /= p_1[2]
    print(Vector(p_1))
    
    #print(Vector(p_1)[0], Vector(p_1)[1])
    
    p_inv = np.linalg.inv(np.vstack((P, np.ones(4)))) @ p_1
    print(Vector(p_inv))
    print(co)

    
    #Camera_Cls.Save_Data({'Path':'./Documents', 'Name':'Test_Image_10'})
    #print(f'[INFO]Result in interatin {i}: {res}')
    
    #print(f'Result: {np.round(np.array(Camera_Cls.K()), 5)}')
if __name__ == '__main__':
    main()

