import cv2
import numpy as np
from PIL import Image

# https://alimustoofaa.medium.com/how-to-load-model-yolov8-onnx-cv2-dnn-3e176cde16e6
# https://github.com/Alimustoofaa/yolov8-custom-dataset/blob/main/YoloV8_Train_Detection.ipynb

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# https://alimustoofaa.medium.com/how-to-load-model-yolov8-onnx-cv2-dnn-3e176cde16e6
# Load Model
net = cv2.dnn.readNet('yolov8s_custom.onnx')

# https://www.thepythoncode.com/code/yolo-object-detection-with-opencv-and-pytorch-in-python?utm_content=cmp-true
# https://mkadric.medium.com/how-to-use-yolo-object-detection-model-1604cf9bbaed
# https://gilberttanner.com/blog/yolo-object-detection-with-opencv/
# https://datagen.tech/guides/face-recognition/face-detection-with-opencv-2-quick-tutorials/
# https://dontrepeatyourself.org/post/object-detection-with-python-deep-learning-and-opencv/
# https://github.com/opencv/opencv/issues/19252
# https://medium.com/@vsreedharachari/practical-implementation-of-object-detection-on-image-with-opencv-and-yolo-v3-pre-trained-weights-d65d6c13401b
# https://thinkinfi.com/yolo-object-detection-using-python-opencv/
# https://stackoverflow.com/questions/59004046/saving-bounding-box-coordinates-and-images-based-on-the-class-type-into-a-differ


INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4

# load the image from disk
image = cv2.imread('Image_00031.png')

blob = cv2.dnn.blobFromImage(image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)

net.setInput(blob)
preds = net.forward()
preds = preds.transpose((0, 2, 1))

# Extract output detection
class_ids, confs, boxes = list(), list(), list()

image_height, image_width, _ = image.shape
x_factor = image_width / INPUT_WIDTH
y_factor = image_height / INPUT_HEIGHT

rows = preds[0].shape[0]

for i in range(rows):
    row = preds[0][i]
    conf = row[4]
    
    classes_score = row[4:]
    _,_,_, max_idx = cv2.minMaxLoc(classes_score)
    class_id = max_idx[1]
    #print(classes_score[class_id])
    if (classes_score[class_id] > .5):
        confs.append(conf)
        #label = CLASESS_YOLO[int(class_id)]
        #class_ids.append(label)

        #extract boxes
        x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item() 
        left = int((x - 0.5 * w) * x_factor)
        top = int((y - 0.5 * h) * y_factor)
        width = int(w * x_factor)
        height = int(h * y_factor)
        box = np.array([left, top, width, height])
        boxes.append(box)
        
r_class_ids, r_confs, r_boxes = list(), list(), list()

indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.25, 0.45) 
for i in indexes:
    #print(class_ids[i])
    #r_class_ids.append(class_ids[i])
    r_confs.append(confs[i])
    r_boxes.append(boxes[i])

for i in indexes:
    box = boxes[i]
    left = box[0]
    top = box[1]
    width = box[2]
    height = box[3]
    
    cv2.rectangle(image, (left, top), (left + width, top + height), (0,255,0), 3)

# Displays the image in the window.
cv2.imshow('Synthetic Data Generated by Blender', image)
cv2.waitKey(0)
cv2.destroyAllWindows()