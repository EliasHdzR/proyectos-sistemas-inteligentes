"""
https://stackoverflow.com/questions/72851576/corner-detection-in-opencv
"""

import numpy as np
import cv2 as cv
import cv2
import math


# function to compute distance between 2 points
def distance(pt1, pt2):
    (x1, y1), (x2, y2) = pt1, pt2
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


# img = cv.imread('resources/Triangulo.jpg')
img = cv2.imread('resources/BlobTest.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# gray = cv2.blur(gray, (3, 3))
gray = np.float32(gray)

dst = cv.cornerHarris(gray, 2, 3, 0.04)
# dst = cv.cornerHarris(gray,5,3,0.04)

# result is dilated for marking the corners, not important
dst = cv.dilate(dst, None)

# visualize the corners
mask = np.zeros_like(gray)
mask[dst > 0.01 * dst.max()] = 255

# storing coordinate positions of all points in a list
coordinates = np.argwhere(mask)
coor_list = coordinates.tolist()

# points beyond this threshold are preserved
thresh = 20

# Threshold for an optimal value, it may vary depending on the image (Original Algorithm)
# draw final corners
img3 = img.copy()
img2 = img.copy()
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

print(len(coor_list))
cv.imshow('dst', img)

# if cv.waitKey(0) & 0xff == 27:
#    exit()


#
coor_list_2 = coor_list.copy()

# iterate for every 2 points
i = 1
for pt1 in coor_list:
    for pt2 in coor_list[i::1]:
        #        if(distance(pt1, pt2) < thresh):
        if (False):
            # to avoid removing a point if already removed
            try:
                coor_list_2.remove(pt2)
            except:
                pass
    i += 1

for pt in coor_list_2:
    img2 = cv2.circle(img2, tuple(reversed(pt)), 3, (0, 0, 255), -1)

print(len(coor_list_2))
cv.imshow('dst2', img2)

gray = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)
# find Harris corners
gray = np.float32(gray)
dst = cv.cornerHarris(gray, 2, 3, 0.04)
dst = cv.dilate(dst, None)
ret, dst = cv.threshold(dst, 0.01 * dst.max(), 255, 0)
dst = np.uint8(dst)

# find centroids
ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)

# define the criteria to stop and refine the corners
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

# Now draw them
res = np.hstack((centroids, corners))
res = np.intp(res)
print(len(res))
# print (centroids)
# print (corners)

# img3[res[:,1],res[:,0]]=[0,0,255]
# img3[res[:,3],res[:,2]] = [0,255,0]

for pt in corners:
    img3 = cv2.circle(img3, tuple(int(num) for num in pt), 3, (255, 0, 255), -1)

# cv.imwrite('subpixel5.png',img)
cv.imshow('dst3', img3)

if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()