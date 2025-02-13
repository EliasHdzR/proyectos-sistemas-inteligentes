# import the cv2 library
import cv2
 
# The function cv2.imread() is used to read an image.
img_grayscale = cv2.imread('resources/test.jpg',0)
img_color = cv2.imread('resources/test.jpg')
 
# The function cv2.imshow() is used to display an image in a window.
cv2.imshow('graycsale image',img_grayscale)
cv2.imshow('color image',img_color)
 
# waitKey() waits for a key press to close the window and 0 specifies indefinite loop
cv2.imwrite('grayscale.png',img_grayscale)
cv2.waitKey(0)
 
# cv2.destroyAllWindows() simply destroys all the windows we created.
cv2.destroyAllWindows()
 
# The function cv2.imwrite() is used to write an image.

