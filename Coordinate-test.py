import sys
import os
import cv2
import time
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized
def test():
    print('test')
    source='0'
    dataset = LoadStreams('0', img_size=640)
    update=1
    coor = [[527, 389],
            [665, 398],
            [840, 407],]
    coor1 = [
        [518,395],
        [665,398],
        [839,404]
    ]
    coor2=[
        [495,429],
        [712,416],
        [1005,407]
    ]
    while True:


        path, img, im0s, img_depth, depth, sgyro, intrin, vid_cap = next(dataset)
        print('保存图片')
        cv2.imwrite('./a-color.jpg', im0s[0])
        if update==1:

            ppx = intrin[0]
            ppy = intrin[1]
            fx = intrin[2]
            fy = intrin[3]
            update=0


        #pixel_x, pixel_y=650,360
        for co in coor2:
            pixel_x, pixel_y=co
            print(pixel_x, pixel_y)

            z = depth.get_distance(pixel_x, pixel_y)
            x, y = [(pixel_x - ppx) * z / fx, (pixel_y - ppy) * z / fy]
            print('像素坐标：（{},{}）实际坐标（mm）：({:.3f},{:.3f},{:.3f})'.format(
                 pixel_x, pixel_y, x * 1000, y * 1000, z * 1000
            ))
        time.sleep(1)

if __name__=='__main__':
    test()