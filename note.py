import cv2
import numpy as np
import math
import os

a= np.loadtxt("../exp/d/dep860.txt",delimiter=",")
rows, cols = a.shape

#print(a,type(a),a.shape,a.dtype,a.size)
salice = a[int(rows/2-rows/10):int(rows/2+rows/10),int(cols/2-cols/10):int(cols/2+cols/10)]

mid = int(salice.mean())
max = a.max()
b = a / max
img = (b * 255).astype("uint8")

img = cv2.medianBlur(img, 25)
cv2.imshow('img',img)
print(mid)
mask_low = a < 0.95*mid
mask_high = a > 1.05*mid
print(mask_high.shape,type(mask_low))
img_mask = img*mask_high
cv2.imshow('mask',img_mask)
img_mask = cv2.medianBlur(img_mask, 15)
ret, thresh = cv2.threshold(img_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(thresh)
#img_mask = cv2.medianBlur(img, 25)

cv2.imshow('mask_high',img_mask)
cv2.imshow('thresh',thresh)
# cv2.imwrite("../exp/d/mask935.jpg",thresh)
# np.savetxt("../exp/d/mask935.txt", thresh, fmt="%d", delimiter=",")


cv2.waitKey()