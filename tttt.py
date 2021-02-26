import numpy as np
import cv2
import math
def cal_angel(img):
    #img = cv2.imread('/home/limeng/Pictures/data/4.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 中值滤波
    blur = cv2.medianBlur(gray, 3)  # 模板大小3*3

    # blur = cv2.GaussianBlur(gray, (5,5), 0)
    # 二值化
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow('thresh',thresh)

    contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours.sort(key=lambda contours:len(contours), reverse=True)
    print('con',contours)



    temp = np.ones(thresh.shape, np.uint8) * 255
    # 画出轮廓：temp是白色幕布，contours是轮廓，-1表示全画，然后是颜色，厚度
    for i in range(4):
        pass

    cv2.drawContours(temp, contours, 8, (0, 0, 255), 2)
    cv2.imshow('con',temp)
    minLineLength = 20
    maxLineGap = 15
    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 80, minLineLength, maxLineGap)
    print('直线数量', len(lines))
    x1, y1, x2, y2 = 0, 0, 0, 0
    lens = 0
    for i in range(5):
        for _x1, _y1, _x2, _y2 in lines[i]:
            _x = abs(_x1 - _x2)
            _y = abs(_y1 - _y2)
            _len = _y ** 2 + _x ** 2
            if (_len > lens):
                lens = _len
                x1, y1, x2, y2 = _x1, _y1, _x2, _y2
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
    thela = math.atan((y2 - y1) / (x2 - x1))
    print('直线角度', thela)
    #return thela
    #cv2.imshow("contours", thresh)
    cv2.line(thresh, (x1, y1), (x2, y2), (0, 255, 0), 3)
    # print('直线角度',)
    #cv2.imshow("contours-line", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__=='__main__':
    img = cv2.imread('/home/limeng/Pictures/data/33.png')
    cal_angel(img)

# #读取图片
# img = cv2.imread('/home/limeng/Pictures/data/66.png')
# # height, width = img.shape[0:2]
# # for i in range(width):
# #     img[0, i] = [255]*3
# #     img[height-1, i] = [255]*3
# # for j in range(height):
# #     img[j, 0] = [255]*3
# #     img[j, width-1] = [255]*3
# #
# # # 去掉灰色线（即噪声）
# # for i in range(height):
# #     for j in range(width):
# #         if list(img[i,j]) == [204,213,204]:
# #             img[i,j]=[255]*3
#
# # 把图片转换为灰度模式
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # 中值滤波
# blur = cv2.medianBlur(gray, 3)  # 模板大小3*3
#
# #blur = cv2.GaussianBlur(gray, (5,5), 0)
# # 二值化
# ret,thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
# binaryImg=thresh
# #二值化，canny检测
# #binaryImg = cv2.Canny(thresh,50,200)
#
# #寻找轮廓
# #也可以这么写：
# #binary,contours, hierarchy = cv2.findContours(binaryImg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# #这样，可以直接用contours表示
# cv2.imshow("binary", binaryImg)
# h = cv2.findContours(binaryImg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
# cv2.imshow("bina_ry", binaryImg)
# print('是否相等',)
# #提取轮廓
# contours = h[0]
# # temp = np.ones(binaryImg.shape,np.uint8)*255
# # cv2.drawContours(temp,contours,-1,(0,255,0),3)
# # h = cv2.findContours(temp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
# contours = h[0]
# #打印返回值，这是一个元组
# print(type(h))
# #打印轮廓类型，这是个列表
# print(type(h[1]))
# #查看轮廓数量
# print (len(contours))
#
# #创建白色幕布
# temp = np.ones(binaryImg.shape,np.uint8)*255
# #画出轮廓：temp是白色幕布，contours是轮廓，-1表示全画，然后是颜色，厚度
# cv2.drawContours(temp,contours,-1,(0,0,255),2)
# cv2.imshow('temp',temp)
# minLineLength = 20
# maxLineGap = 15
# lines = cv2.HoughLinesP(temp, 1, np.pi / 180, 80, minLineLength, maxLineGap)
# print('直线数量',len(lines))
# x1, y1, x2, y2=0,0,0,0
# len=0
# for i in range(5):
#     for _x1, _y1, _x2, _y2 in lines[i]:
#         _x=abs(_x1-_x2)
#         _y=abs(_y1-_y2)
#         _len=_y**2+_x**2
#         if (_len>len):
#             len=_len
#             x1, y1, x2, y2=_x1, _y1, _x2, _y2
# cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
# thela= math.atan((y2-y1)/(x2-x1))
# print('直线角度',thela)
# # print('直线个数',len(lines))
# # x1, y1, x2, y2=0,0,0,0
# # len=0
# # for line in lines:
# #     for _x1, _y1, _x2, _y2 in line:
# #         _x=abs(_x1-_x2)
# #         _y=abs(_y1-_y2)
# #         _len=_y**2+_x**2
# #         if (_len>len):
# #             len=_len
# #             x1, y1, x2, y2=_x1, _y1, _x2, _y2
# # print('最长直线',len)
# cv2.imshow("contours",temp)
# cv2.line(temp, (x1, y1), (x2, y2), (0, 255, 0), 3)
#         #print('直线角度',)
# cv2.imshow("contours-line",temp)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
