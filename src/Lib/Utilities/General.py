# Typing (Support for type hints)
import typing as tp

# https://christianbernecker.medium.com/convert-bounding-boxes-from-coco-to-pascal-voc-to-yolo-and-back-660dc6178742
# https://towardsdatascience.com/convert-pascal-voc-xml-to-yolo-for-object-detection-f969811ccba5

def __YOLO_to_PASCAL_VOC(Bounding_Box: tp.Tuple[float], Resolution: tp.Tuple[int, int]) -> tp.Tuple[int]:
    aux_width = Bounding_Box['width'] * float(Resolution['x']); aux_height = Bounding_Box['height'] * float(Resolution['y'])

    x_min = ((2.0 * Bounding_Box['x_c'] * float(Resolution['x'])) - aux_width)/2.0
    y_min = ((2.0 * Bounding_Box['y_c'] * float(Resolution['y'])) - aux_height)/2.0

    return {'x_min': int(x_min), 'y_min': int(y_min), 'x_max': int(x_min + aux_width), 'y_max': int(y_min + aux_height)}

def __PASCAL_VOC_to_YOLO(Bounding_Box: tp.Tuple[int], Resolution: tp.Tuple[int, int]) -> tp.Tuple[float]:
    return {'x_c': (Bounding_Box['x_max'] + Bounding_Box['x_min'])/(2.0*float(Resolution['x'])), 
            'y_c': (Bounding_Box['y_max'] + Bounding_Box['y_min'])/(2.0*float(Resolution['y'])), 
            'width': (Bounding_Box['x_max'] - Bounding_Box['x_min'])/float(Resolution['x']), 
            'height': (Bounding_Box['y_max'] - Bounding_Box['y_min'])/float(Resolution['y'])}

def Convert_Annotation(format_in: str, format_out: str, Bounding_Box: tp.Tuple[tp.Union[int, float]], Resolution: tp.Tuple[int, int]) -> tp.Tuple[tp.Union[int, float]]:
    try:
        assert format_in in ['YOLO', 'PASCAL_VOC'] and format_out in ['YOLO', 'PASCAL_VOC'] and format_in != format_out

        if format_in == 'YOLO':
            return __YOLO_to_PASCAL_VOC(Bounding_Box, Resolution)
        
        if format_in == 'PASCAL_VOC':
            return __PASCAL_VOC_to_YOLO(Bounding_Box, Resolution)
        
    except AssertionError as error:
        print(f'[ERROR] Information: {error}')
    