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
            img = cv2.imread(path)
            font = cv2.FONT_HERSHEY_DUPLEX

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.medianBlur(gray, 3)  # 模板大小3*3

            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = cv2.putText(thresh, str(ret), (30, 30), font, 1.2, (0, 0, 255), 1)

            blur = cv2.medianBlur(thresh, 5)  # 模板大小3*3
            thresh = ~blur
            rows, cols = img.shape[:2]
            print(rows, cols)
            moment = cv2.moments(blur)
            # print(moment)
            # area = rows * cols - moment['m00'] / 255
            info = ''
            # width = area / rows
            # info += str(int(width)) + ' '
            # w_x, w_y = moment['m10'] / moment['m00'], moment['m01'] / moment['m00']
            w_x, w_y = moment['m10'] / moment['m00'], moment['m01'] / moment['m00']
            print(w_x, w_y)

            a = moment['m20'] / moment['m00'] - w_x * w_x
            b = moment['m11'] / moment['m00'] - w_x * w_y
            c = moment['m02'] / moment['m00'] - w_y * w_y
            theta = cv2.fastAtan2(2 * b, (a - c)) / 2


            half_rows = int(rows / 2)
            if w_y > half_rows:
                blur = blur[half_rows:, :]
            else:
                blur = blur[:half_rows, :]
            moment = cv2.moments(blur)
            w_x, w_y = moment['m10'] / moment['m00'], moment['m01'] / moment['m00']
            area = half_rows * cols - moment['m00'] / 255
            width = area / half_rows
            info += str(int(width)) + ' '
            info += str(int(theta))

            # 图片对象、文本、像素、字体、字体大小、颜色、字体粗细

            # img = cv2.putText(img, res, (1100, 1164), font, 5.5, (0, 0, 255), 2, )
            img = cv2.putText(img, info, (10, 30), font, 0.7, (0, 0, 255), 1)
            save_path = '../ddd/' + path[-9:]
            print(save_path)
            cv2.imwrite(save_path, img)


def cal_moments(path):
    print(path[-9:])
    font = cv2.FONT_HERSHEY_DUPLEX  # 设置字体
    img = cv2.imread(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)  # 模板大小3*3

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print(thresh)
    thresh = cv2.putText(thresh, str(ret), (30, 30), font, 1.2, (0, 0, 255), 1)

    blur = cv2.medianBlur(thresh, 5)  # 模板大小3*3
    print(blur.shape)
    # thresh = ~blur

    # contours, hierarchy = cv2.findContours(thresh, 1, 2)
    #
    # cnt = contours[7]

    # M = cv2.moments(cnt)
    # print(M)
    # rows, cols = img.shape[:2]
    # [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    # lefty = int((-x * vy / vx) + y)
    # righty = int(((cols - x) * vy / vx) + y)
    # img = cv2.line(img, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)

    # rect = cv2.minAreaRect(cnt)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # img = cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    #
    # cv2.imshow("img", img)

    # #moment = M
    #
    #
    #
    rows, cols = img.shape[:2]
    print(rows, cols)
    moment = cv2.moments(blur)
    # print(moment)
    # area = rows * cols - moment['m00'] / 255
    # #area = moment['m00'] / 255
    info = ''
    # width = area / rows

    w_x, w_y = moment['m10'] / moment['m00'], moment['m01'] / moment['m00']
    print(w_x, w_y)
    half_rows = int(rows / 2)
    if w_y > half_rows:
        blur = blur[half_rows:, :]
    else:
        blur = blur[:half_rows, :]
    moment = cv2.moments(blur)
    w_x, w_y = moment['m10'] / moment['m00'], moment['m01'] / moment['m00']
    area = half_rows * cols - moment['m00'] / 255
    width = area / half_rows
    info += str(int(width)) + ' '

    a = moment['m20'] / moment['m00'] - w_x * w_x
    b = moment['m11'] / moment['m00'] - w_x * w_y
    c = moment['m02'] / moment['m00'] - w_y * w_y
    theta = cv2.fastAtan2(2 * b, (a - c)) / 2

    info += str(int(theta))

    # 图片对象、文本、像素、字体、字体大小、颜色、字体粗细
    # img = cv2.putText(img, res, (1100, 1164), font, 5.5, (0, 0, 255), 2, )
    img = cv2.putText(img, info, (10, 30), font, 0.7, (0, 0, 255), 1)

    #

    # print(theta)
    #
    #
    # #theta = fastAtan2(2 * b, (a - c)) / 2; /
    # print(moment)
    cv2.imshow("gray", gray)
    cv2.imshow("blur", blur)
    cv2.imshow("thresh", thresh)

    cv2.imshow("result", img)
    cv2.waitKey(0)
    return img


if __name__ == "__main__":
    #cal_moments('../aaa/pipe5.jpg')
    bianli('../aaa')
