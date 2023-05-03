import cv2
import numpy as np

net = cv2.dnn.readNet('yolov8s_custom.onnx')
image = cv2.imread('Image_00031.png')

# Inside Function ...
image_height, image_width, _ = image.shape
x_factor = image_width / 640
y_factor = image_height / 640

blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True, crop=False)
net.setInput(blob)

#ln = net.getLayerNames()
#ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

output = net.forward()
output = output.transpose((0, 2, 1))


# loop over the number of detected objects
for detection in output[0, :, :]: # output[0, 0, :, :] has a shape of: (100, 7)
    # the confidence of the model regarding the detected object
    probability = detection[4]

    # if the confidence of the model is lower than 50%,
    # we do nothing (continue looping)
    if probability < 0.5:
        continue

    _,_,_, max_idx = cv2.minMaxLoc(detection[4:])

    # perform element-wise multiplication to get
    # the (x, y) coordinates of the bounding box
    x, y, w, h = detection[0].item(), detection[1].item(), detection[2].item(), detection[3].item() 
    left = int((x - 0.5 * w) * x_factor)
    top = int((y - 0.5 * h) * y_factor)
    width = int(w * x_factor)
    height = int(h * y_factor)

    # draw the bounding box of the object
    cv2.rectangle(image, (left, top), (left + width, top + height), (0, 255, 0), thickness=2)

    # extract the ID of the detected object to get its name
    class_id = int(detection[1])


cv2.imshow('Image', image)
cv2.waitKey()
#outs = net.forward(output_layers)

#preds = net.forward()
#preds = preds.transpose((0, 2, 1))