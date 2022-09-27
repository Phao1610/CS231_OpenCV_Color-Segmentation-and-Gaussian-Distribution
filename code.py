from unittest import result
import cv2
from cv2 import threshold
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

cropping = False

x_start, y_start, x_end, y_end = 0, 0, 0, 0

image = cv2.imread('images.jfif')
oriImage = image.copy()

lst_rois = []
refPt = []

def mouse_crop(event, x, y, flags, param):
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping

    # if the left mouse button was DOWN, start RECORDING
    # (x, y) coordinates and indicate that cropping is being
    if event == cv2.EVENT_LBUTTONDOWN:
        x_start, y_start, x_end, y_end = x, y, x, y
        cropping = True

    # Mouse is Moving
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            x_end, y_end = x, y

    # if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates
        x_end, y_end = x, y
        cropping = False # cropping is finished

        refPoint = [(x_start, y_start), (x_end, y_end)]

        if len(refPoint) == 2: #when two points were found
            roi = oriImage[refPoint[0][1]:refPoint[1][1], refPoint[0][0]:refPoint[1][0]]
            cv2.imshow("Cropped", roi)
            sel = np.average(roi,axis = (0,1))
            lst_rois.append(roi)
            print(sel)

cv2.namedWindow("image")
cv2.setMouseCallback("image", mouse_crop)

while True:

    i = image.copy()

    if not cropping:
        cv2.imshow("image", image)

    elif cropping:
        cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
        cv2.imshow("image", i)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("r"):
        image = clone.copy()
    # if the 'c' key is pressed, break from the loop
    elif key == ord("c"):
        break
    cv2.waitKey(1)

# close all open windows
cv2.destroyAllWindows()

def avg(lst_rois):
    k = 2
    x = [[] for i in range(4)]
    for roi in lst_rois:
        x[0] += roi[:,:,0].flatten().tolist()
        x[1] += roi[:,:,1].flatten().tolist()
        x[2] += roi[:,:,2].flatten().tolist()

    avg_roi = np.array([
        np.mean(np.array(x[0])),
        np.mean(np.array(x[1])),
        np.mean(np.array(x[2]))
        ])

    var_roi = np.array([
        np.var(np.array(x[0])),
        np.var(np.array(x[1])),
        np.var(np.array(x[2]))
        ])

    sigma = np.sqrt(var_roi)

    x[3].append(avg_roi-k*sigma)
    x[3].append(avg_roi+k*sigma)   
    
    return x[3]


print(avg(lst_rois))

img = image.copy()
thresh = avg(lst_rois)
img[cv2.inRange(img, thresh[0], thresh[1]) != 0] = np.array([0, 0, 0])

bd = cv2.imread("th.jpg")
bd = cv2.resize(bd, (img.shape[1], img.shape[0]))
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i,j].all() == 0:
            img[i,j] = bd[i,j]




cv2.imshow('img',img)

cv2.waitKey(0)


cv2.imwrite('result.jpg',img)

