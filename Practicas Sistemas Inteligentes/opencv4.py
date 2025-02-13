# Import packages
import cv2
import numpy as np
 
img = cv2.imread('resources/test.jpg', 0)
print(img.shape) # Print image shape
cv2.imshow("original", img)
 
# Cropping an image
#[start_row:end_row, start_col:end_col]
cropped_image = img[80:280, 150:330]
print(cropped_image.shape)

# Display cropped image
cv2.imshow("cropped", cropped_image)
 
# Save the cropped image
cv2.imwrite("resources/cropped_image.jpg", cropped_image)
 
cv2.waitKey(0)
cv2.destroyAllWindows()
