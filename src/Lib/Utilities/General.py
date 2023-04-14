# Typing (Support for type hints)
import typing as tp

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

def Convert_Annotation(format_in: str, format_out: str, Bounding_Box: tp.Tuple[tp.List[tp.Union[int, float]]], Resolution: tp.Tuple[int, int]) -> tp.Tuple[tp.Union[int, float]]:
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
    