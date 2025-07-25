"""
## =========================================================================== ## 
MIT License
Copyright (c) 2023 Roman Parak
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
## =========================================================================== ## 
Author   : Roman Parak
Email    : Roman.Parak@outlook.com
Github   : https://github.com/rparak
File Name: General.py
## =========================================================================== ## 
"""

# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp

def Get_Min_Max(vertices: tp.List[float]) -> tp.Tuple[tp.List[float], tp.List[float]]:
    """
    Description:
        Get the minimum and maximum X, Y, Z values of the input vertices.

    Args:
        (1) vertices [Vector<float> 8x3]: Vertices of the bounding box (AABB, OBB).

    Returns:
        (1) parameter [Vector<float> 1x3]: Minimum X, Y, Z values of the input vertices.
        (2) parameter [Vector<float> 1x3]: Maximum X, Y, Z values of the input vertices.
    """

    min_vec3 = np.array([vertices[0, 0], vertices[0, 1], vertices[0, 2]], dtype=np.float64)
    max_vec3 = min_vec3.copy()
    
    for _, verts_i in enumerate(vertices[1::]):
        for j, verts_ij in enumerate(verts_i):
            if verts_ij < min_vec3[j]:
                min_vec3[j] = verts_ij

            if verts_ij > max_vec3[j]:
                max_vec3[j] = verts_ij
                
    return (min_vec3, max_vec3)

def __YOLO_to_PASCAL_VOC(Bounding_Box: tp.Tuple[tp.List[float]], Resolution: tp.Tuple[int, int]) -> tp.Tuple[tp.List[int]]:
    """
    Description:
        Function to convert the bounding box from YOLO format to PASCAL VOC format.

    Args:
        (1) Bounding_Box [Dictionary {'x_c': float, 'y_c': float, 'width': float, 'height': float}]: Input bounding box in the YOLO format.
        (2) Resolution [Dictionary {'x': width, 'y': height}]: Resolution of the processed image.

    Returns:
        (1) parameter [Dictionary {'x_min': int, 'y_min': int, 'x_max': int, 'y_max': int}]: Output bounding box in the PASCAL_VOC 
                                                                                             format.
    """

    aux_width = Bounding_Box['width'] * float(Resolution['x']); aux_height = Bounding_Box['height'] * float(Resolution['y'])

    x_min = ((2.0 * Bounding_Box['x_c'] * float(Resolution['x'])) - aux_width)/2.0
    y_min = ((2.0 * Bounding_Box['y_c'] * float(Resolution['y'])) - aux_height)/2.0

    return {'x_min': int(x_min), 'y_min': int(y_min), 'x_max': int(x_min + aux_width), 'y_max': int(y_min + aux_height)}

def __PASCAL_VOC_to_YOLO(Bounding_Box: tp.Tuple[tp.List[int]], Resolution: tp.Tuple[int, int]) -> tp.Tuple[tp.List[float]]:
    """
    Description:
        Function to convert the bounding box from PASCAL VOC format to YOLO format.

    Args:
        (1) Bounding_Box [Dictionary {'x_min': int, 'y_min': int, 'x_max': int, 'y_max': int}]: Input bounding box in the PASCAL VOC 
                                                                                                format.
        (2) Resolution [Dictionary {'x': width, 'y': height}]: Resolution of the processed image.

    Returns:
        (1) parameter [Dictionary {'x_c': float, 'y_c': float, 'width': float, 'height': float}]: Output bounding box in the YOLO format.
    """
        
    return {'x_c': (Bounding_Box['x_max'] + Bounding_Box['x_min'])/(2.0*float(Resolution['x'])), 
            'y_c': (Bounding_Box['y_max'] + Bounding_Box['y_min'])/(2.0*float(Resolution['y'])), 
            'width': (Bounding_Box['x_max'] - Bounding_Box['x_min'])/float(Resolution['x']), 
            'height': (Bounding_Box['y_max'] - Bounding_Box['y_min'])/float(Resolution['y'])}

def Convert_Boundig_Box_Data(format_in: str, format_out: str, Bounding_Box: tp.Tuple[tp.List[tp.Union[int, float]]], Resolution: tp.Tuple[int, int]) -> tp.Tuple[tp.List[tp.Union[int, float]]]:
    """
    Description:
        Function to convert the bounding box data from one format to another.

            Available conversions:
                YOLO -> PASCAL VOC, PASCAL VOC -> YOLO

            Input dictionary of individual formats:
                YOLO = {'x_c': float, 'y_c': float, 'width': float, 'height': float}
                PASCAL_VOC = {'x_min': int, 'y_min': int, 'x_max': int, 'y_max': int}

            PASCAL VOC:
                The {x_min} and {y_min} are the coordinates of the left top corner and {x_max} and {y_max} 
                are the coordinates of right bottom corner of the bounding box.

            YOLO:
                The {x_c} and {y_c} are the normalized coordinates of the center of the bounding box. The {width} 
                and {height} are normalized lengths.
        
    Args:
        (1, 2) format_in, format_out [string]: Input and output format conversion.
        (3) Bounding_Box [Dictionary .. See notes at the top.]: Input bounding box in the specified format.
        (4) Resolution [Dictionary {'x': width, 'y': height}]: Resolution of the processed image.
                                                        
    Returns:
        (1) parameter [Dictionary .. See notes at the top.]: Output bounding box in the desired format.
    """
    
    try:
        assert format_in in ['YOLO', 'PASCAL_VOC'] and format_out in ['YOLO', 'PASCAL_VOC'] and format_in != format_out

        if format_in == 'YOLO':
            return __YOLO_to_PASCAL_VOC(Bounding_Box, Resolution)
        
        if format_in == 'PASCAL_VOC':
            return __PASCAL_VOC_to_YOLO(Bounding_Box, Resolution)
        
    except AssertionError as error:
        print(f'[ERROR] Information: {error}')
        print('[ERROR] Invalid input parameters. The input/output format must be named YOLO or PASCAL_VOC \
              and cannot be the same..')
        
def Get_2D_Coordinates_Bounding_Box(vertices: tp.List[tp.List[float]], P: tp.List[tp.List[float]], Resolution: tp.Tuple[int, int], format_out: str) -> tp.Tuple[tp.List[float]]:
    """
    Description:
        Get the 2D coordinates of the bounding box from the rendered object scanned by the camera.

        The pinhole camera model describes the mathematical relationship between the coordinates 
        of a point in three-dimensional space and its projection.

            Equation:
                s[u, v, 1]{3, 1} = P{3, 4} x [X, Y, Z, 1]{4, 1},
                
                where {P} is the projection matrix of the camera, {u, v} are the x and y coordinates of the pixel in the camera, {X, Y, Z} are the coordinates 
                of a 3D point in the world coordinate space
    Args:
        (1) vertices [Matrix<float> 3xn]: The vertices of the scanned object.
                                          Note:
                                            Where n is the number of vertices.
        (2) P [Matrix<float> 3x4]: Projection matrix of the camera.
        (3) Resolution [Dictionary {'x': width, 'y': height}]: Resolution of the processed image.
        (4) format_out [string]: The format into which the data is to be converted.

    Returns:
        (1) parameter [Dictionary .. See the Convert_Boundig_Box_Data() function.]: Output bounding box in the desired format.
    """

    try:
        assert format_out == 'YOLO'

        # Extension of the matrix {P{3, 4}} to a square matrix {P{4, 4}}.
        P_extended = np.vstack((P, np.ones(4)))

        # Projection mapping from world coordinates to pixel coordinates.
        p = []
        for _, verts_i in enumerate(vertices):
            p_tmp = (P_extended @ np.hstack((np.array(verts_i), 1)))[0:-1]
            # By dividing the z-coordinate of the camera relative to the world origin, the theoretical 
            # value of the pixel coordinates can be found.
            p.append(p_tmp/p_tmp[-1])

        # Get the minimum and maximum values of the pixel coordinates.
        (p_min, p_max) = Get_Min_Max(np.array(p, dtype=np.float64))

        return Convert_Boundig_Box_Data('PASCAL_VOC', format_out, {'x_min': p_min[0], 'y_min': p_min[1], 'x_max': p_max[0], 'y_max': p_max[1]}, 
                                        Resolution)
    
    except AssertionError as error:
        print(f'[ERROR] Information: {error}')
        print('[INFO] The output format must be YOLO, as other formats are not yet implemented.')
    