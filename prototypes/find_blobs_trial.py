__author__ = 'diana'

import numpy as np
import cv2


img = cv2.imread('/home/diana/Desktop/targets_example.png', cv2.IMREAD_GRAYSCALE)

params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0;
params.maxThreshold = 255;

# Filter by Area.
params.filterByArea = True
params.minArea = 600

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.5
#
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.5
#
# # Filter by Inertia
# params.filterByInertia = True
# params.minInertiaRatio = 0.01


detector = cv2.SimpleBlobDetector(params)
keypoints = detector.detect(img)

# Detect blobs.
keypoints = detector.detect(img)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)


cv2.imshow('My image', img)
cv2.waitKey(0)