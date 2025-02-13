# let's start with the Imports 
import cv2
import numpy as np
 
# Read the image using imread function
image = cv2.imread('resources/vocho.jpg')
print(image.shape)
cv2.imshow('Original Image', image)
 
# let's downscale the image using new  width and height
down_width = 100
down_height = 100
down_points = (down_width, down_height)
resized_down = cv2.resize(image, down_points, interpolation= cv2.INTER_LINEAR)
print(resized_down.shape)
cv2.imwrite('resources/vocho_chico.png', resized_down)

# let's downscale the image using new  width and height
down_width = 1024
down_height = 353
down_points = (down_width, down_height)
resized_down2 = cv2.resize(resized_down, down_points, interpolation= cv2.INTER_LINEAR)
print(resized_down.shape)
cv2.imwrite('resources/vocho_chico2.png', resized_down2)
 
# let's upscale the image using new  width and height
up_width = 600
up_height = 400
up_points = (up_width, up_height)
resized_up = cv2.resize(image, up_points, interpolation= cv2.INTER_LINEAR)
 
# Display images
cv2.imshow('Resized Down by defining height and width', resized_down)
cv2.waitKey()
cv2.imshow('Resized Down by defining height and width', resized_down2)
cv2.waitKey()
cv2.imshow('Resized Up image by defining height and width', resized_up)
cv2.waitKey()
 
#press any key to close the windows
cv2.destroyAllWindows()
