# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# OpenCV (Computer Vision) [pip3 install opencv-python]
import cv2
# Custom Library:
#   ../Lib/Utilities/General
import Lib.Utilities.General

def Draw_Bounding_Box(image: tp.List[tp.List[int]], bounding_box_properties: tp.Tuple[str, str, tp.List[tp.Union[int, float]]], format: str, Color: tp.List[int], 
                      fill_box: bool, show_info: bool) -> tp.List[tp.List[int]]:
    """
    Description:
        Function to draw the bounding box of an object with additional dependencies (name, accuracy, etc.) in the raw image.

    Args:
        (1) image [Vector<float> Image Shape {Resolution<x, y>}]: Input raw image.
        (2) bounding_box_properties [Dictionary {'Name': string, 'Accuracy', string, 
                                                 'Data': Vector<int/float> 1x4}]: Bounding box properties.
        (3) format [string]: The format of the bounding box input data. Available formats: YOLO, Pascal_VOC.
        (4) Color [Vector<float> 1x3]: Color of the box and other dependencies.
        (5) fill_box [bool]: Information about whether or not to fill the rectangle.
        (6) show_info [bool]: Information about whether or not to show additional text.

    Returns:
        (1) parameter [Vector<float> Image Shape {Resolution<x, y>}]: Output image extended with bounding box and other dependencies.

    Example:
        image_out = Draw_Bounding_Box(image, bounding_box_properties = {'Name': 'Obj_Name_Id_0', 'Accuracy': '100', 'Data': None}, format = 'YOLO/Pascal_VOC', 
                                      Color = (0, 255, 0), fill_box = False, show_info = False)
    """

    image_out = image.copy()

    # Set the properties of the drawing bounding box.
    #   Image Resolution: [x: Height, y: Width]
    Resolution = {'x': image_out.shape[0], 'y': image_out.shape[1]}
    #   Line width of the rectangle.
    line_width = 1
    # Offset of an additional rectangles.
    offset = 5
    
 
    # Obtain data in PASCAL_VOC format to determine the bounding box to be rendered.
    #   data = {'x_min', 'y_min', 'x_max', 'y_max'}
    if format == 'YOLO':
        data = Lib.Utilities.General.Convert_Annotation(format, 'PASCAL_VOC', bounding_box_properties['Data'], Resolution) 
    data = bounding_box_properties['Data']

    x_min = data['x_min']; y_min = data['y_min']
    x_max = data['x_max']; y_max = data['y_max']
    box_w = x_max - x_min; box_h = y_max - y_min

    if fill_box == True:
        # Transparency coefficient.
        alpha = 0.10

        # The main rectangle that bounds the object.
        cv2.rectangle(image_out, (x_min, y_min), (x_max, y_max), Color, -1)

        if show_info == True:
            # Additional rectangles that bounds information about the object.
            cv2.rectangle(image_out, (x_min, y_min - (int(box_h/4.0) + offset)), (x_min + box_w, y_min - offset), 
                          Color, -1)
            cv2.rectangle(image_out, (x_max + offset, y_min - (int(box_h/4.0) + offset)), (x_max + offset + int(box_w/2.0), y_min - offset), 
                          Color, -1)
            
        image_out = cv2.addWeighted(image_out, alpha, image, 1 - alpha, 0)

    # Draw a modified rectangle around the object.
    #  --          --
    # |              |    
    #      Object    
    # |              |
    #  --          --
    #   Corner: Left Top
    cv2.line(image_out, (x_min, y_min), (x_min + int(box_w/4.0), y_min), Color, line_width)
    cv2.line(image_out, (x_min, y_min), (x_min, y_min + int(box_h/4.0)), Color, line_width)
    #   Corner: Left Bottom
    cv2.line(image_out, (x_min, y_max), (x_min + int(box_w/4.0), y_max), Color, line_width)
    cv2.line(image_out, (x_min, y_max), (x_min, y_max - int(box_h/4.0)), Color, line_width)
    #   Corner: Right Top
    cv2.line(image_out, (x_max, y_min), (x_max - int(box_w/4.0), y_min), Color, line_width)
    cv2.line(image_out, (x_max, y_min), (x_max, y_min + int(box_h/4.0)), Color, line_width)
    #   Corner: Right Bottom
    cv2.line(image_out, (x_max, y_max), (x_max - int(box_w/4.0), y_max), Color, line_width)
    cv2.line(image_out, (x_max, y_max), (x_max, y_max - int(box_h/4.0)), Color, line_width)

    if show_info == True:
         # The font of the text shown in the image.
        txt_font = cv2.FONT_HERSHEY_SIMPLEX

        # A rectangle with the name of the object.
        cv2.rectangle(image_out, (x_min, y_min - (int(box_h/4.0) + offset)), (x_min + box_w, y_min - offset), 
                      Color, line_width)
        #   Get the text boundary with the object name.
        #       Parameters: [0.5: font_scale, 1: thickness]
        txt_name_boundary = cv2.getTextSize(bounding_box_properties['Name'], txt_font, 0.5, line_width)[0]

        # Get the coefficient of the displacement difference between the rectangles.
        #   Rectangle Id: Name
        f = np.array([box_w/2.0, int(box_h/4.0)/2]) - np.array([txt_name_boundary[0]/2, txt_name_boundary[1]/2])
        cv2.putText(image_out, bounding_box_properties['Name'], (x_min + int(f[0]), (y_min - offset) - int(f[1])), txt_font, 0.5, Color, line_width)

        # A rectangle indicating the accuracy of the match.
        cv2.rectangle(image_out, (x_max + offset, y_min - (int(box_h/4.0) + offset)), (x_max + offset + int(box_w/2.0), y_min - offset), 
                      Color, line_width)
        # For accuracy, we use the same method as for the name.
        txt_name_boundary = cv2.getTextSize(bounding_box_properties['Accuracy'] + ' %', txt_font, 0.5, line_width)[0]
        f = np.array([int(box_w/2.0)/2.0, int(box_h/4.0)/2]) - np.array([txt_name_boundary[0]/2, txt_name_boundary[1]/2])
        cv2.putText(image_out, bounding_box_properties['Accuracy'] + ' %', (x_max + offset + int(f[0]), (y_min - offset) - int(f[1])), txt_font, 0.5, Color, line_width)

    return image_out