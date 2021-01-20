# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np

path = '/home/limeng/Pictures/data/'

def pre_pce(imagepath):
    print('正在处理',imagepath)

    img = cv2.imread(imagepath, 1)

    # 将图片的边缘变为白色
    height, width = img.shape[0:2]
    for i in range(width):
        img[0, i] = [255]*3
        img[height-1, i] = [255]*3
    for j in range(height):
        img[j, 0] = [255]*3
        img[j, width-1] = [255]*3

    # 去掉灰色线（即噪声）
    for i in range(height):
        for j in range(width):
            if list(img[i,j]) == [204,213,204]:
                img[i,j]=[255]*3

    # 把图片转换为灰度模式
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 中值滤波
    blur = cv2.medianBlur(gray, 3)  # 模板大小3*3
    # 二值化
    ret,thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    a = cv2.findContours(thresh, 2, 2)
    print(len(a))
    #cal_angle(thresh)
    # 保存图片

    save_path=path[:-4]+'_pre.png'
    cv2.imwrite(save_path, thresh)
    #print(save_path)
    #cal_angle(save_path)
# -*- coding: utf-8 -*-

def cal_angle(img):

    print('处理角度')
    contours, hierarchy= cv2.findContours(img, 2, 2)


    # print('contours',contours)
    # print('hierarchy',hierarchy)

    for cnt in contours:

        # 最小外界矩形的宽度和高度
        width, height = cv2.minAreaRect(cnt)[1]

        if width* height > 100:
            # 最小的外接矩形
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点
            box = np.int0(box)

            if 0 not in box.ravel():

                '''绘制最小外界矩形
                for i in range(4):
                    cv2.line(image, tuple(box[i]), tuple(box[(i+1)%4]), 0)  # 5
                '''
                # 旋转角度
                theta = cv2.minAreaRect(cnt)[2]
                if abs(theta) <= 45:
                    print('图片的旋转角度为%s.'%theta)
                    angle = theta

    # 仿射变换,对图片旋转angle角度
    h, w = img.shape
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # 保存旋转后的图片
    #cv2.imwrite('F://CHN_Char/after_rotated.png', rotated)
    cv2.imwrite(path[:-4]+'_cal.png', rotated)



for root, dirs, files in os.walk(path):
    print(root)
    for file in files:
        path=root+file
        if path.find('pre') > 0:
            continue
        if path.find('cal') > 0:
            continue
        pre_pce(path)




