import cv2
import numpy as np
import os
import csv
import time
import math

image = cv2.imread("ss.jpg")
dim = (3*image.shape[0],3*image.shape[1])
image1 = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
gray = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray.jpg",gray)
edge_result = cv2.Canny(gray,150,250)
cv2.imwrite("edges.jpg",edge_result)
laplacian = cv2.Laplacian(gray,cv2.CV_64F)
cv2.imwrite("laplacian.jpg",laplacian)
sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
cv2.imwrite("sobelx.jpg",sobelx)
sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
cv2.imwrite("sobely.jpg",sobely)
sobel = sobelx + sobely
cv2.imwrite("sobel.jpg",sobel)
fin = laplacian+gray
cv2.imwrite("fin.jpg",fin)
