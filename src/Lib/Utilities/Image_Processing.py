# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Typing (Support for type hints)
import typing as tp
# OpenCV (Computer Vision) [pip3 install opencv-python]
import cv2
# Custom Library:
#   ../Lib/Utilities/General
import Lib.Utilities.General
#   ../Lib/Transformation/Utilities/Mathematics
import Lib.Transformation.Utilities.Mathematics as Mathematics

def Get_Alpha_Beta_Parameters(image: tp.List[tp.List[int]], clip_limit: float) -> tp.Tuple[float, float]:
    """
    Description:
        Function to adjust the contrast and brightness parameters of the input image by clipping the histogram.

        The main core of the function is to obtain the alpha and beta parameters 
        to determine the equation:
            g(i, j) = alpha * f(i, j) + beta,

            where g(i, j) are the input (source) pixels of the image, f(i, j) are the output pixels 
            of the image, and alpha, beta are the contrast and brightness parameters.

    Args:
        (1) image [Vector<float> Image Shape {Resolution<x, y>}]: Input raw image.
        (2) clip_limit [float]: Parameter for histogram clipping in percentage.

    Returns:
        (1) parameter [float]: A gain (contrast) parameter called alpha.
        (2) parameter [float]: A bias (brightness) parameter called beta.
    """

    image_copy = image.copy()

    # Calculate the grayscale histogram of the image.
    image_hist = cv2.calcHist([image_copy], [0], None, [256], [0, 256])

    # Get the cumulative sum of the elements (histogram).
    c_hist = np.cumsum(image_hist)

    # Modify the percentage to clip the histogram.
    c_hist_max = c_hist[-1]
    clip_limit_mod = clip_limit * (c_hist_max/float(100.0*2.0))

    # Clip the histogram if the values are outside the limit.
    min_value = 0; max_value = c_hist.size - 1
    for _, c_hist_i in enumerate(c_hist):
        if c_hist_i < clip_limit_mod:
            min_value += 1

        if c_hist_i >= (c_hist_max - clip_limit_mod):
            max_value -= 1

    # Express the alpha and beta parameters.
    #   Gain (contrast) parameter.
    alpha = 255 / (max_value - min_value)
    #   Bias (brightness) parameter.
    beta  = (-1) * (min_value * alpha)

    return (alpha, beta)

def Draw_Bounding_Box(image: tp.List[tp.List[int]], bounding_box_properties: tp.Tuple[str, str, tp.List[tp.Union[int, float]]], format: str, Color: tp.List[int], 
                      fill_box: bool, show_info: bool) -> tp.List[tp.List[int]]:
    """
    Description:
        Function to draw the bounding box of an object with additional dependencies (name, precision, etc.) in the raw image.

    Args:
        (1) image [Vector<float> Image Shape {Resolution<x, y>}]: Input raw image.
        (2) bounding_box_properties [Dictionary {'Name': string, 'Precision', string, 
                                                 'Data': Vector<int/float> 1x4}]: Bounding box properties.
        (3) format [string]: The format of the bounding box input data. Available formats: YOLO, Pascal_VOC.
        (4) Color [Vector<float> 1x3]: Color of the box and other dependencies.
        (5) fill_box [bool]: Information about whether or not to fill the rectangle.
        (6) show_info [bool]: Information about whether or not to show additional text.

    Returns:
        (1) parameter [Vector<float> Image Shape {Resolution<x, y>}]: Output image extended with bounding box and other dependencies.

    Example:
        image_out = Draw_Bounding_Box(image, bounding_box_properties = {'Name': 'Obj_Name_Id_0', 'Precision': '100', 'Data': None}, format = 'YOLO/Pascal_VOC', 
                                      Color = (0, 255, 0), fill_box = False, show_info = False)
    """

    image_out = image.copy()

    # Set the properties of the drawing bounding box.
    #   Image Resolution: [x: Height, y: Width]
    Resolution = {'x': image_out.shape[1], 'y': image_out.shape[0]}
    #   Line width of the rectangle.
    line_width = 2
    # Offset of an additional rectangles.
    offset = 5
    
    # Obtain data in PASCAL_VOC format to determine the bounding box to be rendered.
    #   data = {'x_min', 'y_min', 'x_max', 'y_max'}
    if format == 'YOLO':
        data = Lib.Utilities.General.Convert_Boundig_Box_Data(format, 'PASCAL_VOC', bounding_box_properties['Data'], Resolution)
    elif format == 'PASCAL_VOC':
        data = bounding_box_properties['Data']

    x_min = data['x_min']; y_min = data['y_min']
    x_max = data['x_max']; y_max = data['y_max']
    box_w = x_max - x_min; box_h = y_max - y_min

    # Fill the box with the desired transparency coefficient.
    if fill_box == True:
        # Transparency coefficient.
        alpha = 0.10

        # The main rectangle that bounds the object.
        cv2.rectangle(image_out, (x_min, y_min), (x_max, y_max), Color, -1)
            
        # Change from one image to another. To blend the image, add weights that determine 
        # the transparency and translucency of the images.
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
        # Additional rectangles that bounds information about the object.
        cv2.rectangle(image_out, (x_min, y_min - (int(box_h/4.0) + offset)), (x_min + box_w, y_min - offset), 
                        Color, -1)
        cv2.rectangle(image_out, (x_max + offset, y_min - (int(box_h/4.0) + offset)), (x_max + offset + int(box_w/2.0), y_min - offset), 
                        Color, -1)
        
        # The font of the text shown in the image.
        txt_font = cv2.FONT_HERSHEY_SIMPLEX

        # A rectangle with the name of the object.
        cv2.rectangle(image_out, (x_min, y_min - (int(box_h/4.0) + offset)), (x_min + box_w, y_min - offset), 
                      Color, line_width)
        
        #   Get the text boundary with the object name.
        #       Parameters: [0.5: font_scale, 1: thickness]
        txt_name_boundary = cv2.getTextSize(bounding_box_properties['Name'], txt_font, 0.55, int(line_width/2))[0]

        # Get the coefficient of the displacement difference between the rectangles.
        #   Rectangle Id: Name
        f = np.array([box_w/2.0, int(box_h/4.0)/2]) - np.array([txt_name_boundary[0]/2, txt_name_boundary[1]/2])
        cv2.putText(image_out, bounding_box_properties['Name'], (x_min + int(f[0]), (y_min - offset) - int(f[1])), txt_font, 0.55, (0, 0, 0), int(line_width/2), cv2.LINE_AA)

        # A rectangle indicating the precision of the match.
        cv2.rectangle(image_out, (x_max + offset, y_min - (int(box_h/4.0) + offset)), (x_max + offset + int(box_w/2.0), y_min - offset), 
                      Color, line_width)
        # For precision, we use the same method as for the name.
        txt_name_boundary = cv2.getTextSize(bounding_box_properties['Precision'], txt_font, 0.55, int(line_width/2))[0]
        f = np.array([int(box_w/2.0)/2.0, int(box_h/4.0)/2]) - np.array([txt_name_boundary[0]/2, txt_name_boundary[1]/2])
        cv2.putText(image_out, bounding_box_properties['Precision'], (x_max + offset + int(f[0]), (y_min - offset) - int(f[1])), txt_font, 0.55, (0, 0, 0), int(line_width/2), cv2.LINE_AA)

    return image_out

def YOLO_Object_Detection(image: tp.List[tp.List[int]], model_onnx: str, image_size: int, confidence: float):
    """
    Description:
        Function for object detection using the trained YOLO model. The model in our case must be in *.onnx format, converted 
        from the official *.pt model.

        More information about training, validation, etc. of the model can be found here:
            ../Lib/YOLO/YOLOv8_Train_Custom_Dataset.ipynb

    Args:
        (1) image [Vector<float> Image Shape {Resolution<x, y>}]: Input image to be used for object detection.
        (2) model_onnx [str]: Input model in *.onnx format.
                              Note:
                                More information about the onnx format can be found at: https://onnx.ai
        (3) image_size [int]: Image size as scalar. The size must match the size of the image when training the model.
        (4) confidence [float]: The required minimum object confidence threshold for detection.
    
    Returns:
        (1) parameter [int or Vector<int> 1xn]: The class ID of the detected object.
        (2) parameter [float or Vector<float> 1xn]: The actual object confidence threshold for detection.
        (3) parameter [Dictionary {'x_min': int, 'y_min': int, 
                                   'x_max': int, 'y_max': int} x n]: Bounding box in the PASCAL VOC format 
                                                                     on the actual image size.
        Note:
            Where n is the number of detected objects.
    """

    # Parameters:
    #   Image Resolution: [x: Height, y: Width]
    Resolution = {'x': image.shape[1], 'y': image.shape[0]}
    #   The coefficient (factor) of the processed image.
    Image_Coeff = {'x': Resolution['x'] / image_size, 
                   'y': Resolution['y'] / image_size}
    #   Threshold for applying non-maximum suppression.
    CONST_THRESHOLD = 0.5
    
    # Create a blob from the input image.
    blob = cv2.dnn.blobFromImage(image, 1.0/255.0, (image_size, image_size), swapRB=True, crop=False)

    # Get the names of the output layers.
    layer_names = model_onnx.getLayerNames()
    layer_names = [layer_names[i - 1] for i in model_onnx.getUnconnectedOutLayers()]

    # Perform a forward pass through the YOLO object detector to obtain bounding 
    # boxes and associated probabilities.
    model_onnx.setInput(blob)
    output_layers = model_onnx.forward(layer_names)

    # Find the bounding boxes that correspond to the input object detection parameters.
    class_ids = []; bounding_boxes = []; confidences = []
    for _, output_layers_i in enumerate(output_layers):
        for _, output_layers_ij in enumerate(output_layers_i.T):
            # Extract the class identification number (class id) and the confidence 
            # of the actual object detection.
            scores_tmp = output_layers_ij[4:]; class_id_tmp = Mathematics.Max(scores_tmp)[0]
            confidence_tmp = scores_tmp[class_id_tmp]

            # Consider only predictions that are higher than the desired confidence value specified 
            # by the function input.
            if confidence_tmp > confidence:
                # Add the class id and confidence to the list. 
                class_ids.append(class_id_tmp); confidences.append(confidence_tmp[0])
                # Extract the coordinates (YOLO) of the bounding box.
                bounding_box_tmp = output_layers_ij[0:4].reshape(1, 4)[0]
                #   Convert the coordinates of the bounding box to the desired format.
                x = int((bounding_box_tmp[0] - 0.5 * bounding_box_tmp[2]) * Image_Coeff['x'])
                y = int((bounding_box_tmp[1] - 0.5 * bounding_box_tmp[3]) * Image_Coeff['y'])
                w = int(bounding_box_tmp[2] * Image_Coeff['x'])
                h = int(bounding_box_tmp[3] * Image_Coeff['y'])
                # Add a bounding box to the list.
                bounding_boxes.append([x, y, w, h])

    # Perform a non-maximal suppression relative to the previously defined score.
    indexes = cv2.dnn.NMSBoxes(bounding_boxes, confidences, confidence, CONST_THRESHOLD)
    
    # At least one detection should be successful, otherwise just report a failed detection.
    if isinstance(indexes, np.ndarray):
        print(f'[INFO] The model found {indexes.size} object in the input image.')

        # Store parameters over specific indexes.
        class_ids_out = []; bounding_boxes_out = []; confidences_out = []
        for i in indexes.flatten():
            # Extract the class identification number (class id) and the confidence.
            class_ids_out.append(class_ids[i]); confidences_out.append(confidences[i])
            # Extract the bounding box and convert it to PASCAL VOC format.
            bounding_boxes_out.append({'x_min': bounding_boxes[i][0], 
                                       'y_min': bounding_boxes[i][1], 
                                       'x_max': bounding_boxes[i][0] + bounding_boxes[i][2], 
                                       'y_max': bounding_boxes[i][1] + bounding_boxes[i][3]})
            
        return (class_ids_out, bounding_boxes_out, confidences_out)

    else:
        print('[INFO] The model did not find object in the input image.')
        return (None, None, None)

    