import sys
import cv2
import numpy as np
# https://christianbernecker.medium.com/convert-bounding-boxes-from-coco-to-pascal-voc-to-yolo-and-back-660dc6178742

def Draw_Bounding_Box(image, Bounding_Box = {'Name': 'Obj_Name_Id_0', 'Accuracy': '100', 'Data': None}, format = 'YOLO/Pascal_VOC', Resolution = {'x': 2064, 'y': 1544}, 
                      Color = (0, 255, 0)):
    pass

def pascal_voc_to_yolo(x1, y1, x2, y2, image_w, image_h):
    return [((x2 + x1)/(2*image_w)), ((y2 + y1)/(2*image_h)), (x2 - x1)/image_w, (y2 - y1)/image_h]

def yolo_to_pascal_voc(x_center, y_center, w, h,  image_w, image_h):
    w = w * image_w
    h = h * image_h
    x1 = ((2 * x_center * image_w) - w)/2
    y1 = ((2 * y_center * image_h) - h)/2
    x2 = x1 + w
    y2 = y1 + h
    return [x1, y1, x2, y2]

def MyRec(rgb,x,y,w,h,v=20,color=(200,0,0),thikness =1):
    """To draw stylish rectangle around the objects"""
    cv2.line(rgb, (x,y),(x+v,y), color, thikness)
    cv2.line(rgb, (x,y),(x,y+v), color, thikness)

    cv2.line(rgb, (x+w,y),(x+w-v,y), color, thikness)
    cv2.line(rgb, (x+w,y),(x+w,y+v), color, thikness)

    cv2.line(rgb, (x,y+h),(x,y+h-v), color, thikness)
    cv2.line(rgb, (x,y+h),(x+v,y+h), color, thikness)

    cv2.line(rgb, (x+w,y+h),(x+w,y+h-v), color, thikness)
    cv2.line(rgb, (x+w,y+h),(x+w-v,y+h), color, thikness)

def main():
    img = cv2.imread('Test_Image_10.png')
    overlay = img.copy()

    RGB_Color = (0,255,0)

    height = overlay.shape[0]; width = overlay.shape[1]

    XML = {'x_min': 950, 'y_min': 614, 'x_max': 1132, 'y_max': 752}
    YOLO = {'x_c': 0.504347, 'y_c': 0.442419, 'w': 0.087728, 'h': 0.089276}

    #print(pascal_voc_to_yolo(XML['x_min'], XML['y_min'], XML['x_max'], XML['y_max'], width, height))
    #print(yolo_to_pascal_voc(YOLO['x_c'], YOLO['y_c'], YOLO['w'], YOLO['h'], width, height))

    x, y, w, h = XML['x_min'], XML['y_min'], (XML['x_max'] - XML['x_min']), (XML['y_max'] - XML['y_min'])

    cv2.rectangle(overlay,(x,y-35),(x+w,y-5),RGB_Color, -1)
    cv2.rectangle(overlay,(XML['x_max'] + 5,y-35),(XML['x_max'] + 5 + 80,y-5),RGB_Color, -1)
    cv2.rectangle(overlay,(x,y),(x+w,y+h),RGB_Color, -1)

    alpha = 0.15  # Transparency factor.
    # Following line overlays transparent rectangle over the image
    image_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # setup text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "T_Joint"
    # get boundary of this text
    textsize = cv2.getTextSize(text, font, 0.5, 1)[0]
    text_c = np.array([textsize[0]/2, textsize[1]/2])
    rect_c = np.array([w/2.0, 30/2])
    f = rect_c - text_c
    # ...
    cv2.putText(image_new, text, (x + int(f[0]), (y - 5) - int(f[1])), font, 0.5, RGB_Color, 1)

    # setup text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "89.99%"
    # get boundary of this text
    textsize = cv2.getTextSize(text, font, 0.5, 1)[0]
    text_c = np.array([textsize[0]/2, textsize[1]/2])
    rect_c = np.array([80.0/2.0, 30/2])
    f = rect_c - text_c
    # ...
    cv2.putText(image_new, text, (XML['x_max'] + 5 + int(f[0]), (y - 5) - int(f[1])), font, 0.5, RGB_Color, 1)

    MyRec(image_new, x, y, w, h, 20, RGB_Color)
    cv2.rectangle(image_new,(x,y-35),(x+w,y-5),RGB_Color, 1)
    cv2.rectangle(image_new,(XML['x_max'] + 5,y-35),(XML['x_max'] + 5 + 80,y-5),RGB_Color, 1)

    cv2.imshow('Test OpenCV', image_new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main())