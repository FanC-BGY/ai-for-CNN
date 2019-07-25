#! usr/bin/env python
import cv2
import numpy as np

path=r'D:\groceries\AI\ai-for-CNN\img\dragon-mother.jpg'
img=cv2.imread(path)
print(img.type)
# check the values of Gaussion kernel
kernel=cv2.getGaussianKernel(7,5)
print(kernel)

# gaussian blur
g1_img=cv2.GaussianBlur(img,(7,7),5)
g2_img=cv2.sepFilter2D(img,-1,kernel,kernel)


# 2nd -derivative
kernel_lap = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.float32)
lap_img = cv2.filter2D(img, -1, kernel_lap)


sift = cv2.xfeatures2d.SIFT_create()
# detect SIFT
kp = sift.detect(img,None)   # None for mask
# compute SIFT descriptor
kp,des = sift.compute(img,kp)
print(des.shape)
img_sift= cv2.drawKeypoints(img,kp,outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# show the pictures
# cv2.imshow('dragon-mother',img)
# cv2.imshow('blur1',g1_img)
# cv2.imshow('blur2',g2_img)
# cv2.imshow('lplace',lap_img)
# cv2.imshow('sift',img_sift)
#
# key=cv2.waitKey()
# if 27==key:
#     cv2.destroyAllWindows()
