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
    #cv2.imshow('img', img)
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
            delta = (y2 - y1) / (x2 - x1)
            _thela = math.atan(delta)
            thela.append(_thela)
            # a = np.cos(_thela)
            # b = np.sin(_thela)
            #
            # x1 = int(x1 + 1000 * (-b))
            # y1 = int(y1 + 1000 * (a))
            # x2 = int(x2 - 1000 * (-b))
            # y2 = int(y2 - 1000 * (a))
            cv2.line(img, (x1, y1), (x2+1000, y2+int(1000*delta)), (0, 0, 255), 2)
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

def cal_angel(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 中值滤波
    blur = cv2.medianBlur(gray, 3)  # 模板大小3*3
    # blur = cv2.GaussianBlur(gray, (5,5), 0)
    # 二值化
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    # 膨胀
    dilate = cv2.dilate(thresh, kernel2)
    # 腐蚀
    dst = cv2.erode(dilate, kernel)
    # dst = cv2.dilate(dst, kernel)
    edges = cv2.Canny(dst, ret - 10, ret + 10)
    try:
        minLineLength = 200
        maxLineGap = 15
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
        # lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        print('直线数量', len(lines))
    except TypeError:
        minLineLength = 20
        maxLineGap = 10
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength, maxLineGap)
        # lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    thela = []
    try:
        for i in range(len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                # if x1 == x2:
                #     _thela = 0
                _thela = math.atan((y2 - y1) / (x2 - x1 + 1))
                thela.append(_thela)

        mode, counts = find_mode(thela)
        return mode, counts
    except TypeError:
        return 99,99



def fenge():
    img = cv2.imread('/home/sz2/aaa/pipe_84.jpg')  # 直接读为灰度图像
    #   缩小图像10倍(因为我的图片太大，所以要缩小10倍方便看看效果)
    # height, width = img.shape[:2]
    # size = (int(width * 0.1), int(height * 0.1))  # bgr
    # img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    # BGR转化为HSV
    HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow("imageHSV", HSV)
    cv2.imshow('image', img)
    color = [
        ([0, 0, 0], [90, 90, 90])  # 蓝色范围~这个是我自己试验的范围，可根据实际情况自行调整~注意：数值按[b,g,r]排布
    ]
    # 如果color中定义了几种颜色区间，都可以分割出来
    for (lower, upper) in color:
        # 创建NumPy数组
        lower = np.array(lower, dtype="uint8")  # 颜色下限
        upper = np.array(upper, dtype="uint8")  # 颜色上限
        # 根据阈值找到对应颜色
        mask = cv2.inRange(HSV, lower, upper)  # 查找处于范围区间的
        #mask = 255 - mask  # 留下铝材区域
        output = cv2.bitwise_and(img, img, mask=mask)  # 获取铝材区域
        output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)

        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        # 中值滤波
        blur = cv2.medianBlur(gray, 3)  # 模板大小3*3

        # blur = cv2.GaussianBlur(gray, (5,5), 0)
        # 二值化
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow("thresh",thresh)



        #bgroutput = cv2.cvtColor(output,cv2.COLOR_HSV2BGR)
        # 展示图片
        cv2.imshow("images", np.hstack([img, output]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(mask.shape)
        print(mask[0])
        print(len(contours))
        cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
        for i in contours:
            print(cv2.contourArea(i))  # 计算缺陷区域面积
            x, y, w, h = cv2.boundingRect(i)  # 画矩形框
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # cv.imwrite(show_result_path, match_img_color)
        cv2.imshow("detect", img)
        cv2.imshow("chanle", img)

    cv2.waitKey(0)


if __name__ == "__main__":
    img = cv2.imread('pics/7.png')
    func(img)
    #fenge()
