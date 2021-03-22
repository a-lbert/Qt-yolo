# USAGE
# python static_saliency.py --image images/neymar.jpg

# import the necessary packages
#import argparse
import cv2
import numpy as np
import math
import os
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
def saliency(path):
	image = cv2.imread(path)


	saliency = cv2.saliency.StaticSaliencyFineGrained_create()
	(success, saliencyMap) = saliency.computeSaliency(image)
	# saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
	# (success, saliencyMap) = saliency.computeSaliency(image)
	# print(type(saliencyMap))
	# print(np.median(saliencyMap))
	#
	# print(saliencyMap > np.median(saliencyMap))
	median = np.median(saliencyMap)
	_saliencyMap =  saliencyMap <  0.25 * median
	#_saliencyMap_ = saliencyMap > 0.2 * median
	#print(type(_saliencyMap))
	saliencyMap = (_saliencyMap * 255).astype("uint8")

	_, labels, stats, centroids = cv2.connectedComponentsWithStats(saliencyMap)
	print(len(stats))
	for i in range(len(stats)):
		print(i,stats[i])
	print(type(stats))
	mask=~(labels == 166)
	mask2=~(labels == 124)
	saliencyMap = saliencyMap*mask*mask2
	print(stats)
	#saliencyMap[84:84+67,212:212+30] = 0
	#print(saliencyMap == (_saliencyMap*_saliencyMap_ * 255).astype("uint8"))



	# if we would like a *binary* map that we could process for contours,
	# compute convex hull's, extract bounding boxes, etc., we can
	# additionally threshold the saliency map
	ret, threshMap = cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	edges = cv2.Canny(threshMap, ret - 10, ret + 10)
	cv2.imshow('edges', edges)
	lines = cv2.HoughLines(threshMap, 1, np.pi / 180, 10)
	print('直线数量', len(lines))
	thela = []

	for i in range(10):
		r, theta = lines[i][0]
		thela.append(theta)
		if -0.5 < theta < 0.5:

			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a * r
			y0 = b * r
			x1 = int(x0 + 1000 * (-b))
			y1 = int(y0 + 1000 * (a))
			x2 = int(x0 - 1000 * (-b))
			y2 = int(y0 - 1000 * (a))
			cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
			#break
		else:
			#print('pass')
			continue

		# save_path = '/home/sz2/bbb/'+path[-6:]
		# print(save_path)
		# cv2.imwrite(save_path,image)


	cv2.imshow("Thresh", threshMap)

	cv2.imshow("Output", saliencyMap)
	cv2.imshow("Image", image)
	cv2.waitKey(0)

def init():
	# construct the argument parser and parse the arguments
	# ap = argparse.ArgumentParser()
	# ap.add_argument("-i", "--image", required=True,
	# 	help="path to input image")
	# args = vars(ap.parse_args())
	#
	# # args={"image":"images/boat.jpg"}
	#
	# # load the input image
	# image = cv2.imread(args["image"]):
	image = cv2.imread('pics/5.png')
	# # initialize OpenCV's static saliency spectral residual detector and
	# # compute the saliency map
	# saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
	# (success, saliencyMap) = saliency.computeSaliency(image)
	# saliencyMap = (saliencyMap * 255).astype("uint8")
	# cv2.imshow("Image", image)
	# cv2.imshow("Output", saliencyMap)
	# cv2.waitKey(0)

	# initialize OpenCV's static fine grained saliency detector and
	# compute the saliency map
	saliency = cv2.saliency.StaticSaliencyFineGrained_create()
	(success, saliencyMap) = saliency.computeSaliency(image)
	# print(type(saliencyMap))
	# print(np.median(saliencyMap))
	#
	# print(saliencyMap > np.median(saliencyMap))
	_saliencyMap = saliencyMap < 0.25*np.median(saliencyMap)
	saliencyMap = (_saliencyMap * 255).astype("uint8")



	# if we would like a *binary* map that we could process for contours,
	# compute convex hull's, extract bounding boxes, etc., we can
	# additionally threshold the saliency map
	ret,threshMap = cv2.threshold(saliencyMap, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	edges = cv2.Canny(threshMap, ret - 10, ret + 10)
	cv2.imshow('edges', edges)
	lines = cv2.HoughLines(threshMap, 1, np.pi / 180, 10)
	print('直线数量', len(lines))

	for r, theta in lines[i]:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a * r
		y0 = b * r
		x1 = int(x0 + 1000 * (-b))
		y1 = int(y0 + 1000 * (a))
		x2 = int(x0 - 1000 * (-b))
		y2 = int(y0 - 1000 * (a))
		cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)


	cv2.imshow("Output", saliencyMap)
	cv2.imshow("Thresh", threshMap)
	cv2.imshow("Image", image)
	# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	# #膨胀
	# dilate = cv2.dilate(threshMap, kernel)
	# cv2.imshow("dilate", dilate)
	# #腐蚀
	# dst = cv2.erode(dilate, kernel)
	# #dst = cv2.dilate(dst, kernel)
	# cv2.imshow("erode", dst)
	# edges = cv2.Canny(dst, ret - 10, ret + 10)
	# cv2.imshow('edges', edges)
	#
	# minLineLength = 2
	# maxLineGap = 15
	# lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
	# lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)


	#threshMap = cv2.threshold(saliencyMap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
	# contours, _ = cv2.findContours(threshMap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# print(len(contours))
	# cv2.drawContours(threshMap, contours, -1, (255, 0, 0), 1)
	# for i in contours:
	# 	print(cv2.contourArea(i))  # 计算缺陷区域面积
	# 	x, y, w, h = cv2.boundingRect(i)  # 画矩形框
	# 	cv2.rectangle(saliencyMap, (x, y), (x + w, y + h), (0, 255, 0), 1)
	# show the images

	cv2.waitKey(0)

def bianli(file):
	for root, dirs, files in os.walk(file):

		# root 表示当前正在访问的文件夹路径
		# dirs 表示该文件夹下的子目录名list
		# files 表示该文件夹下的文件list

		# 遍历文件
		for f in files:
			path=os.path.join(root, f)
			#print(path)
			#saliency(path)
			cal_moments(path)

def cal_moments(path):
	font = cv2.FONT_HERSHEY_DUPLEX  # 设置字体
	img = cv2.imread(path)
	print(path)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blur = cv2.medianBlur(gray, 3)  # 模板大小3*3
	cv2.imshow("gray",gray)
	ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	thresh = cv2.putText(thresh, str(ret), (30, 30), font, 1.2, (0, 0, 255), 1)
	cv2.imshow("thresh",thresh)
	blur = cv2.medianBlur(thresh, 5)  # 模板大小3*3
	cv2.imshow("blur",blur)
	moment=cv2.moments(blur)
	w_x,w_y = moment['m10']/moment['m00'],moment['m01']/moment['m00']
	a = moment['m20']/moment['m00']-w_x*w_x
	b = moment['m11']/moment['m00']-w_x*w_y
	c = moment['m02']/moment['m00']-w_y*w_y
	theta = cv2.fastAtan2(2*b,(a-c))/2

	res=str(int(theta))

	# 图片对象、文本、像素、字体、字体大小、颜色、字体粗细
	#img = cv2.putText(img, res, (1100, 1164), font, 5.5, (0, 0, 255), 2, )
	img = cv2.putText(img, res, (50, 30), font, 1.2, (0, 0, 255), 1)
	#cv2.imshow("result",img)

	save_path = '../bbb/'+path[-6:]
	print(save_path)
	cv2.imwrite(save_path,img)

	#print(theta)
	#
	#
	# #theta = fastAtan2(2 * b, (a - c)) / 2; /
	#print(moment)
	#cv2.waitKey(0)



if __name__ == "__main__":
    #img = cv2.imread('pics/7.png')

    # func(img)
    bianli('../aaa')
	#saliency('/home/sz2/aaa/pipe1.jpg')
	#saliency('pics/7.png')
	#cal_moments('pics/7.png')