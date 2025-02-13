# Standard imports
import cv2
import numpy as np

Imagen="resources/BlobTest.jpg"
# Imagen="resources/Letras.jpg"
#Imagen="resources/Letras1.png"
#Imagen="resources/Circulos.jpg"
# Imagen="resources/Monedas.png"
#Imagen="resources/Bichos.jpg"

# Read image
imX = cv2.imread(Imagen)
im = cv2.imread(Imagen, cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original", im)

im = cv2.blur(im, (5, 5))
# im = cv2.GaussianBlur(im, (5, 5), 0)
#im = cv2.medianBlur(im, 15)
#im = cv2.bilateralFilter(im, 5, 21, 8)

th, im = cv2.threshold(im,190,255, cv2.THRESH_BINARY)
# cv2.imshow("Thresholded", im)

# Default values of parameters are tuned to extract dark circular blobs.
params = cv2.SimpleBlobDetector_Params()

# Set up the detector with default parameters.
# detector = cv2.SimpleBlobDetector()

# Filter by Area.
params.filterByArea = True
# params.minArea = 5
params.maxArea = 1000000

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.01

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.01

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(params)
# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# im_with_keypoints = im.copy()

ObjetosPosFiltrado = 0
for keyPoint in keypoints:
    print(keyPoint)
    x = keyPoint.pt[0]
    y = keyPoint.pt[1]
    s = keyPoint.size / 2  # Porque este ()
    print(x, y, s)
    if True:
    #if s>7 :
        ObjetosPosFiltrado += 1
            imX = cv2.circle(imX,(int(x),int(y)),int(s), (0, 255, 0),2)
    else:
        print("Discarded!!")

print("Conteo de objetos", len(keypoints))
print("Conteo de objetos PosFiltado", ObjetosPosFiltrado)

# Show keypoints
cv2.imshow("KeypointsA", imX)
cv2.imshow("KeypointsB", im_with_keypoints)
cv2.waitKey(0)
