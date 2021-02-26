import numpy as np
import cv2
import math
#二值化+连通域
#边缘检测+最小外接矩形
def find_mode(thela):

    #print(thela)
    mode = max(thela, key=lambda v: thela.count(v))
    counts = thela.count(mode)
    #限制小数位
    mode = int(mode*100)/100
    #众数个数为1,继续扩展
    if thela.count(mode) == 1:
        # 数据乘10,寻找众数
        thela = list(np.multiply(10, thela))
        thela = list(map(int, thela))
        print('qqqqq',thela)
        mode = max(thela, key=lambda v: thela.count(v))
        #print(mode, thela.count(mode))
        counts = thela.count(mode)
        mode = mode / 10
        #print('aaa')
        if thela.count(mode) == 1:
            thela = list(np.multiply(10, thela))
            mode = max(thela, key=lambda v: thela.count(v))
            print('aaaaaa', thela)
            #print(mode, thela.count(mode))
            counts = thela.count(mode)
            mode = mode / 10
            #print('bbb')
    return mode,counts

def func(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 中值滤波
    blur = cv2.medianBlur(gray, 3)  # 模板大小3*3

    #blur = cv2.GaussianBlur(gray, (5,5), 0)
    # 二值化
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('thresh', thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    #膨胀
    dilate = cv2.dilate(thresh, kernel2)
    cv2.imshow("dilate", dilate)
    #腐蚀
    dst = cv2.erode(dilate, kernel)
    #dst = cv2.dilate(dst, kernel)
    cv2.imshow("erode", dst)
    edges = cv2.Canny(dst, ret - 10, ret + 10)
    cv2.imshow('edges', edges)

    try:
        minLineLength = 200
        maxLineGap = 15
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
        #lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        print('直线数量', len(lines))
    except TypeError:
        minLineLength = 20
        maxLineGap = 10
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength, maxLineGap)
        # lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        print('except——直线数量', len(lines))
    thela = []
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            # if x1 == x2:
            #     _thela = 0
            _thela = math.atan((y2 - y1) / (x2 - x1+1))
            thela.append(_thela)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    mode, counts= find_mode(thela)
    print('返回值',mode,counts)
    cv2.imshow('Result', img)
    # edges = cv2.Canny(dst,ret-10,ret+10)
    # cv2.imshow('edges',edges)
    # contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #
    # temp = np.ones(thresh.shape, np.uint8) * 255
    # cv2.drawContours(temp, contours, -1, (0, 255, 0), 2)
    # cv2.imshow('contours',temp)
    # # con=max(contours, key=lambda s: cv2.contourArea(s))
    # # rect = cv2.minAreaRect(con)
    # # box = np.int0(cv2.boxPoints(rect))
    # # cv2.drawContours(img, [box], 0, (255, 0, 0), 2)
    # # cv2.imshow('rect_max', img)
    # theta = []
    #
    #
    # for i,contour in enumerate(contours):
    #     area = cv2.contourArea(contour)
    #     if area < 2:
    #         print('小于2')
    #         continue
    #     rect = cv2.minAreaRect(contour)
    #     theta.append(int(rect[2]))
    #     box = np.int0(cv2.boxPoints(rect))
    #     cv2.drawContours(img,[box],0,(255,0,0),2)
    #     cv2.imshow('rect',img)
    # print(theta)
    # #print(max(theta, key=lambda v: theta.count(v)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img = cv2.imread('/home/limeng/Pictures/data/9.png')
    func(img)
