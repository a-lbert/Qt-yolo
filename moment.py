import cv2
import numpy as np
import math
import os


def bianli(file):
    for root, dirs, files in os.walk(file):

        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        # 遍历文件
        for f in files:
            path = os.path.join(root, f)
            # print(path)
            # saliency(path)
            #cal_moments(path)
            ceshi(path)
            # img = cv2.imread
            # cal_ang(img)


def cal_angel(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)  # 模板大小3*3
    # cv2.imshow("gray", gray)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # thresh = cv2.putText(thresh, str(ret), (30, 30), font, 1.2, (0, 0, 255), 1)
    # cv2.imshow("thresh", thresh)
    blur = cv2.medianBlur(thresh, 5)  # 模板大小3*3
    cv2.imshow("blur", blur)

    moment = cv2.moments(blur)
    rows, cols = img.shape[:2]
    area = rows * cols - moment['m00'] / 255
    width = area / rows
    w_x, w_y = moment['m10'] / moment['m00'], moment['m01'] / moment['m00']
    a = moment['m20'] / moment['m00'] - w_x * w_x
    b = moment['m11'] / moment['m00'] - w_x * w_y
    c = moment['m02'] / moment['m00'] - w_y * w_y
    theta = cv2.fastAtan2(2 * b, (a - c)) / 2
    # res = str(int(theta)) + '_' + str(int(width))

    # 图片对象、文本、像素、字体、字体大小、颜色、字体粗细
    # img = cv2.putText(img, res, (1100, 1164), font, 5.5, (0, 0, 255), 2, )
    # font = cv2.FONT_HERSHEY_DUPLEX  # 设置字体
    # img = cv2.putText(img, res, (50, 30), font, 1.2, (0, 0, 255), 1)
    # cv2.imshow("result", img)
    # cv2.waitKey(0)
    return theta, width
#截取部分图像->二值化、反转->闭操作->轮廓检测->计算二阶矩->计算宽度（二阶矩面积处以图像高度）以及角度
def cal_ang(img):

    rows, cols = img.shape[:2]
    row = int(rows / 9)
    img = img[1 * row:8 * row, :]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = 255 - thresh
    # cv2.imshow('threddsh', thresh)
    kernel = np.ones((9, 9), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    c_rows, c_cols = closed.shape[:2]

    contours, hierarchy = cv2.findContours(closed, 1, 2)
    contours.sort(key=len, reverse=True)
    cnt = contours[0]
    moment = cv2.moments(cnt)
    width = moment['m00'] / c_rows
    w_x, w_y = moment['m10'] / moment['m00'], moment['m01'] / moment['m00']
    a = moment['m20'] / moment['m00'] - w_x * w_x
    b = moment['m11'] / moment['m00'] - w_x * w_y
    c = moment['m02'] / moment['m00'] - w_y * w_y
    theta = 180 - cv2.fastAtan2(2 * b, (a - c)) / 2
    return theta,width
    # print('theta', theta,width)



def ceshi(path):
    #
    print(path)
    font = cv2.FONT_HERSHEY_DUPLEX  # 设置字体
    img = cv2.imread(path)
    cv2.imshow('im',img)
    rows, cols = img.shape[:2]
    row = int(rows / 9)
    img= img[1 * row:8 * row, :]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh = 255 - thresh

    #cv2.imshow('threddsh', thresh)
    kernel = np.ones((9, 9), np.uint8)
    closed= cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    c_rows, c_cols = closed.shape[:2]
    # blur = cv2.medianBlur(thresh, 5)  # 模板大小3*3
    # cv2.imshow("blur1", blur)
    cv2.imshow("closed", closed)


    # moment = cv2.moments(thresh)
    # rows, cols = thresh.shape[:2]
    # w_y = moment['m01'] / moment['m00']
    # if w_y < rows:
    #     thresh = thresh[:int(rows / 2), :]
    #     cv2.imshow("thresh", thresh)
    # else:
    #     thresh = thresh[int(rows / 2):, :]
    #     cv2.imshow("thresh", thresh)


    contours, hierarchy = cv2.findContours(closed, 1, 2)

    contours.sort(key=len,reverse=True)

    cnt = contours[0]
    moment = cv2.moments(cnt)
    print(c_cols,c_rows)
    area = moment['m00'] / 1

    width = area / c_rows
    print(area,c_rows,width)
    w_x, w_y = moment['m10'] / moment['m00'], moment['m01'] / moment['m00']
    a = moment['m20'] / moment['m00'] - w_x * w_x
    b = moment['m11'] / moment['m00'] - w_x * w_y
    c = moment['m02'] / moment['m00'] - w_y * w_y
    theta = 180-cv2.fastAtan2(2 * b, (a - c)) / 2
    print('theta',theta)
    imag = cv2.drawContours(img, cnt, -1, (255, 0, 0), 1)
    res = str(int(theta))+'_'+str(int(width))
    img = cv2.putText(img, res, (50, 30), font, 1.2, (0, 0, 255), 1)
    save_path = path.replace('250', '2a')


    #cv2.imwrite(save_path, img)
    # cv2.imshow("gray", gray)
    # cv2.imshow('thresh', thresh)
    cv2.imshow("img", img)
    cv2.waitKey(0)


def cal_moments(path):
    font = cv2.FONT_HERSHEY_DUPLEX  # 设置字体
    img = cv2.imread(path)
    print(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)  # 模板大小3*3
    cv2.imshow("gray", gray)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.putText(thresh, str(ret), (30, 30), font, 1.2, (0, 0, 255), 1)
    cv2.imshow("thresh", thresh)
    blur = cv2.medianBlur(thresh, 5)  # 模板大小3*3
    cv2.imshow("blur1", blur)
    moment = cv2.moments(blur)
    w_x, w_y = moment['m10'] / moment['m00'], moment['m01'] / moment['m00']
    a = moment['m20'] / moment['m00'] - w_x * w_x
    b = moment['m11'] / moment['m00'] - w_x * w_y
    c = moment['m02'] / moment['m00'] - w_y * w_y
    theta = cv2.fastAtan2(2 * b, (a - c)) / 2

    print(moment['m10'] / moment['m00'], moment['m01'] / moment['m00'])

    w_y = moment['m01'] / moment['m00']
    print(img.shape[0], img.shape[1], img.shape[2])
    rows = img.shape[0]
    if w_y < rows:
        _blur = blur[:int(rows / 2), :]
        cv2.imshow("blur", _blur)
    else:
        _blur = blur[int(rows / 2):, :]
        cv2.imshow("blur", _blur)

    moment = cv2.moments(_blur)

    rows, cols = _blur.shape[:2]
    print(rows, cols)
    # rows, cols = blur.shape[:2]
    area = rows * cols - moment['m00'] / 255
    width = area / rows

    res = str(int(theta))+'_'+str(int(width))

    # 图片对象、文本、像素、字体、字体大小、颜色、字体粗细
    # img = cv2.putText(img, res, (1100, 1164), font, 5.5, (0, 0, 255), 2, )
    img = cv2.putText(img, res, (50, 30), font, 1.2, (0, 0, 255), 1)
    cv2.imshow("result", img)

    save_path = path.replace('250','b')

    # cv2.imwrite(save_path, img)

    # print(theta)
    #
    #
    # #theta = fastAtan2(2 * b, (a - c)) / 2; /
    # print(moment)
    cv2.waitKey(0)


if __name__ == "__main__":
    # img = cv2.imread('../exp/500/65_92.jpg')
    # cal_angel(img)

    # func(img)
    #bianli('../exp/250/')
    # saliency('/home/sz2/aaa/pipe1.jpg')
    # saliency('pics/7.png')
    #cal_moments('../exp/500/80_95.jpg')
    ceshi('../exp/500/62_93.jpg')
