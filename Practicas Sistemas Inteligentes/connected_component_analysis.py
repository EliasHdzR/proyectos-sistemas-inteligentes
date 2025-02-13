"""
To DOs (Martes)
* Explicar la diferencia entre Blobs vs Connected Components vs Coutours
* https://www.geeksforgeeks.org/line-detection-python-opencv-houghline-method/
* https://www.geeksforgeeks.org/circle-detection-using-opencv-python/
* Procesamiento morfologico
* https://docs.opencv.org/4.x/d4/d8c/tutorial_py_shi_tomasi.html


"""

import cv2
import numpy as np

import random

# Loading the image
img = cv2.imread('BlobTest.jpg')

# preprocess the image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Applying 7x7 Gaussian Blur
blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)

# Applying threshold
threshold = cv2.threshold(blurred, 0, 255,
                          cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# Apply the Component analysis function
analysis = cv2.connectedComponentsWithStats(threshold, 4, cv2.CV_32S)

(totalLabels, label_ids, values, centroid) = analysis

# Initialize a new image to store
# all the output components
output = np.zeros(gray_img.shape, dtype="uint8")
output2 = img.copy()

print(totalLabels)
# Loop through each component
for i in range(1, totalLabels):

    # Area of the component
    area = values[i, cv2.CC_STAT_AREA]

    if (area > 10) and (area < 40000):
        componentMask = (label_ids == i).astype("uint8") * 255
        output2[componentMask > 0] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # color_img = cv2.cvtColor(componentMask, cv2.COLOR_GRAY2RGB)
        # print (color_img[0,0])
        # color_img*=[255,0,0]
        # print (componentMask.shape)
        # print (componentMask.max())
        # print (componentMask.min())

        # cv2.imshow(""+str(i),componentMask)

        # define range of blue color in HSV
        # componentMask = np.zeros_like(img)
        # mask = cv2.fillPoly(componentMask, pts =[big_contours], color=(255,255,255)) # fill the polygon
        # lower_yellow = np.array([15,50,180])
        # upper_yellow = np.array([40,255,255])
        # componentMask = cv2.inRange(img, lower_yellow, upper_yellow)
        # componentMask = (label_ids == i).astype("uint8") * (255, 255, 255)
        # output = cv2.bitwise_or(output, componentMask)

        # output2 =

cv2.imshow("Image", img)
cv2.imshow("Filtered Components", output)
cv2.imshow("Filtered Components", output2)
cv2.waitKey(0)