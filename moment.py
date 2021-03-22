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
			path=os.path.join(root, f)
			#print(path)
			#saliency(path)
			cal_moments(path)
def cal_angel(img):

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blur = cv2.medianBlur(gray, 3)  # 模板大小3*3
	#cv2.imshow("gray", gray)
	ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	#thresh = cv2.putText(thresh, str(ret), (30, 30), font, 1.2, (0, 0, 255), 1)
	#cv2.imshow("thresh", thresh)
	blur = cv2.medianBlur(thresh, 5)  # 模板大小3*3
	#cv2.imshow("blur", blur)
	moment = cv2.moments(blur)
	w_x, w_y = moment['m10'] / moment['m00'], moment['m01'] / moment['m00']
	a = moment['m20'] / moment['m00'] - w_x * w_x
	b = moment['m11'] / moment['m00'] - w_x * w_y
	c = moment['m02'] / moment['m00'] - w_y * w_y
	theta = cv2.fastAtan2(2 * b, (a - c)) / 2
	return theta

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