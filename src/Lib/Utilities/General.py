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

    min_vec3 = np.array([vertices[0, 0], vertices[0, 1], vertices[0, 2]], dtype=np.float32)
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

        # Extension of the matrix {P(3, 4)} to a square matrix {P(4, 4)}.
        P_extended = np.vstack((P, np.ones(4)))

        # ...
        # x = P x X
        # x - 2D image coordinates x[u, v, 1] = P x X[X, Y, Z, 1]
        # x - coordinates of the projection point in pixels
        # x = P x [X, 1]
        # P ...
        # X ... 3D world coordinates 
        # z part of the x must be 1
        # X[X, Y, Z] ... coordinates of a 3D point in the world coordinate space
        # s[u,v,1](3, 1) = P(3, 4) x [X,Y,Z,1](4, 1)

        # https://en.wikipedia.org/wiki/Camera_resectioning
        # https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/agartia3/index.html
        # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

        p = []
        for _, verts_i in enumerate(vertices):
            p_tmp = (P_extended @ np.hstack((np.array(verts_i), 1)))[0:-1]
            p.append(p_tmp/p_tmp[-1])

        # Get the minimum and maximum values of the input pixels.
        (p_min, p_max) = Get_Min_Max(np.array(p, dtype=np.float32))

        return Convert_Boundig_Box_Data('PASCAL_VOC', format_out, {'x_min': p_min[0], 'y_min': p_min[1], 'x_max': p_max[0], 'y_max': p_max[1]}, 
                                        Resolution)

    except AssertionError as error:
        print(f'[ERROR] Information: {error}')
        print('[INFO] The output format must be YOLO, as other formats are not yet implemented.')
    