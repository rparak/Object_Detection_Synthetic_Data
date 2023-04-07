import bpy
import numpy as np

def K(name):
    
    print(bpy.data.cameras[name].sensor_fit)
    scale = bpy.context.scene.render.resolution_percentage / 100.0
    
    # in px ...
    w = bpy.context.scene.render.resolution_x * scale
    h = bpy.context.scene.render.resolution_y * scale
    
    # ...
    print(w, h)
    print(bpy.data.cameras[name].lens)
    print(bpy.data.cameras[name].sensor_width)
    print(bpy.data.cameras[name].sensor_height)
    
    
K('Camera')