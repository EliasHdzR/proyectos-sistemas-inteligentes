    import cv2
import numpy as np

# Let's load a simple image with 3 black squares
Imagen="resources/BlobTest.jpg"
#Imagen="resources/Monedas.png"
#Imagen="resources/Letras.jpg"
#Imagen="resources/Letras1.png"
#Imagen="resources/Bichos.jpg"

imageOr = cv2.imread(Imagen)
image = cv2.imread(Imagen)
#cv2.waitKey(0)

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#gray = cv2.GaussianBlur(gray, (5, 5), 0)
gray  = cv2.bilateralFilter(gray , 5, 21, 7)

# Find Canny edges
edged = cv2.Canny(gray, 30, 200)

#edged = gray
#cv2.waitKey(0)

# Finding Contours
# Use a copy of the image e.g. edged.copy()
# since findContours alters the image
contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#contours, hierarchy = cv2.findContours(edged, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)


for c in contours:
    # calculate moments for each contour
    M = cv2.moments(c)

    # calculate x,y coordinate of center
    if M["m00"] > 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        imageOr = cv2.drawContours(imageOr,[c],0,(128,128,128),2)

        # Centro del Contorno
        #cv2.circle(imageOr, (cX, cY), 5, (0, 0, 255), 5)

        print ("Valid")
        x,y,w,h = cv2.boundingRect(c)
        #imageOr = cv2.rectangle(imageOr,(x,y),(x+w,y+h),(255,0,0),5)

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int_(box)

        #imageOr = cv2.drawContours(imageOr,[box],0,(0,255,255),2)

        area = cv2.contourArea(c)
        #if (True):
        if area > 50:      # Numero arbitrario "Tanteado"
            imageOr = cv2.drawContours(imageOr,[box],0,(255,0,255),2)
            t=0


    else:
        print ("Discarded!")

cv2.imshow('Canny Edges After Contouring - Individual', imageOr)

cv2.imshow('Canny Edges After Contouring', edged)
#cv2.waitKey(0)

print("Number of Contours found = " + str(len(contours)))

# Draw all contours
# -1 signifies drawing all contours
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
