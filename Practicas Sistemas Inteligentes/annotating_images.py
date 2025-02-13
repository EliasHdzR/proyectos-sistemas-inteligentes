# Import dependencies
import cv2
# Read Images
img = cv2.imread('resources/dog.jpg')
# Display Image
cv2.imshow('Original Image',img)
#cv2.waitKey(0)
# Print error message if image is null
if img is None:
    print('Could not read image')
# Draw line on image
imageLine = img.copy()

#Draw the image from point A to B
pointA = (200,80)
pointB = (450,80)
cv2.line(imageLine, pointA, pointB, (255, 255, 0), thickness=3, lineType=cv2.LINE_AA)

# define the center of circle
circle_center = (415,190)
# define the radius of the circle
radius =100
#  Draw a circle using the circle() Function
cv2.circle(imageLine, circle_center, radius, (0, 0, 255), thickness=3, lineType=cv2.LINE_AA) 
# Display the result

circle_center = (215,190)
# define the radius of the circle
radius =100
# draw the filled circle on input image
cv2.circle(imageLine, circle_center, radius, (255, 0, 0), thickness=-1, lineType=cv2.LINE_AA)

start_point =(400,115)
end_point =(575,225)
# draw the rectangle
cv2.rectangle(imageLine, start_point, end_point, (0, 255, 0), thickness= 3, lineType=cv2.LINE_8) 

text = 'I am a Happy dog!'
#org: Where you want to put the text
org = (50,350)
# write the text on the input image
cv2.putText(imageLine, text, org, fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 1.5, color = (250,225,100))

cv2.imshow('Image Line', imageLine)
cv2.waitKey(0)
