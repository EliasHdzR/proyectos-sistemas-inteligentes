# import the necessary packages
import cv2
import numpy as np
import matplotlib.pyplot as plt

# read the image
img = cv2.imread("../resources/vocho.jpg", 0)

# binarize the image
thr, binr = cv2.threshold(img, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

print("umbral obtnenido por OTSU", thr)

# define the kernel
kernel = np.ones((5, 5), np.uint8)

# invert the image
invert = cv2.bitwise_not(binr)

# erode the image
erosion = cv2.erode(invert, kernel,iterations=4)

dilation = cv2.dilate(erosion, kernel,iterations=4	)

# print the output
#plt.imshow(erosion, cmap='gray')
cv2.imshow("Original",img)
cv2.imshow("Umbralizacion",binr)
cv2.imshow("Invertida",invert)
cv2.imshow("Erosionada",erosion)
cv2.imshow("Dilation",dilation)
cv2.waitKey()