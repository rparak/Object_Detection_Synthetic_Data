# System (Default)
import sys
sys.path.append('..')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# OpenCV (Computer Vision) [pip3 install opencv-python]
import cv2
# Custom Library:
#   ../Lib/Utilities/Image_Processing
import Lib.Utilities.Image_Processing

def autoAdjustments(img):
    alow = img.min()
    ahigh = img.max()
    amax = 255
    amin = 0

    print(ahigh, alow)
    # calculate alpha, beta
    alpha = ((amax - amin) / (ahigh - alow))
    beta = amin - alow * alpha
    # perform the operation g(x,y)= α * f(x,y)+ β
    new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return [new_img, alpha, beta]

# Automatic brightness and contrast optimization with optional histogram clipping
def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    print(maximum_gray, minimum_gray)
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return (auto_result, alpha, beta)

def main():
    img_in = cv2.imread('Test_Image_10.png')

    # define the alpha and beta
    alpha = 1.95 # Contrast control
    beta = 0 # Brightness control
    #img_mod = cv2.convertScaleAbs(img_in, alpha=alpha, beta=beta)
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #img_mod = clahe.apply(cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY))
    #img_mod = cv2.equalizeHist(img_in)

    img_mod_1, alpha, beta = automatic_brightness_and_contrast(img_in, 0)
    img_mod_2 = autoAdjustments(img_in)[0]

    #BBox_Properties = {'Name': 'Obj_Name_Id_0', 'Accuracy': '100', 'Data': {'x_min': 950, 'y_min': 614, 'x_max': 1132, 'y_max': 752}}
    #img_out = Lib.Utilities.Image_Processing.Draw_Bounding_Box(img_mod, BBox_Properties, 'PASCAL_VOC', (0, 255, 0), True, True)

    cv2.imshow('Raw', img_in)
    cv2.imshow('Mod 1', img_mod_1)
    cv2.imshow('Mod 2', img_mod_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main())