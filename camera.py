import sys
import os
import cv2

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPalette, QBrush, QPixmap

from PIL import Image
import numpy as np
import cv2
import time
import pyrealsense2 as rs
import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized


# yolo = YOLO()
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
# config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
# profile = pipeline.start(config)
# depth_sensor = profile.get_device().first_depth_sensor()
# depth_scale = depth_sensor.get_depth_scale()
# clipping_distance_in_meters = 1 #1 meter
# clipping_distance = clipping_distance_in_meters / depth_scale
# align_to = rs.stream.color
# align = rs.align(align_to)

class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)

        self.timer_camera = QtCore.QTimer()  # 初始化定时器
        self.set_ui()
        self.slot_init()
        self.__flag_work = 0
        self.x = 0
        self.count = 0
        self.fps = 0.0
        self.is_show_fps = False
        self.show_what_in_pic = False
        self.what_in_pic = ''
        self.ori_num = 0
        self.pro_num = 0
        self.ori = None
        self.pro = None
        self.dataset = None
        self.weights = './szs.pt'
        self.i = 0

    def set_ui(self):
        self.__layout_main = QtWidgets.QHBoxLayout()  # 采用QHBoxLayout类，按照从左到右的顺序来添加控件
        self.__layout_fun_button = QtWidgets.QVBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()  # QVBoxLayout类垂直地摆放小部件

        self.button_open_camera = QtWidgets.QPushButton(u'打开相机')
        self.button_close = QtWidgets.QPushButton(u'退出')
        self.button_showfps = QtWidgets.QPushButton(u'yolov5m')
        self.button_showcount = QtWidgets.QPushButton(u'yolov5l')
        self.button_save_ori = QtWidgets.QPushButton(u'yolov5x')

        # button颜色修改
        button_color = [self.button_open_camera, 
                        self.button_close,
                        self.button_showfps,
                        self.button_showcount,
                        self.button_save_ori,]
        for i in range(len(button_color)):
            button_color[i].setStyleSheet("QPushButton{color:black}"
                                           "QPushButton:hover{color:red}"
                                           "QPushButton{background-color:rgb(78,255,255)}"
                                           "QpushButton{border:2px}"
                                           "QPushButton{border_radius:10px}"
                                           "QPushButton{padding:2px 4px}")

        self.button_open_camera.setMinimumHeight(50)
        self.button_close.setMinimumHeight(50)
        self.button_showfps.setMinimumHeight(10)
        self.button_showcount.setMinimumHeight(10)
        self.button_save_ori.setMinimumHeight(10)
        # 实例化QLabel对象
        self.label = QLabel(self)
        # 设置标签的位置和大小
#        self.label.setGeometry(0, 40, 200, 20)
#        self.label.setScaledContents(False)



        # move()方法是移动窗口在屏幕上的位置到x = 500，y = 500的位置上
        self.move(500, 500)

        # 信息显示
        self.label_show_camera = QtWidgets.QLabel()
        self.label_move = QtWidgets.QLabel()
        self.label_move.setFixedSize(100, 100)

        self.label_show_camera.setFixedSize(1281, 721)
        self.label_show_camera.setAutoFillBackground(False)
        self.label_show_camera.setAlignment(QtCore.Qt.AlignRight)

        self.__layout_fun_button.addWidget(self.button_open_camera)
        self.__layout_fun_button.addWidget(self.button_close)
        self.__layout_fun_button.addWidget(self.button_showfps)
        self.__layout_fun_button.addWidget(self.button_showcount)
        self.__layout_fun_button.addWidget(self.button_save_ori)
        self.__layout_fun_button.addWidget(self.label)
        self.__layout_fun_button.addWidget(self.label_move)

        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addWidget(self.label_show_camera)

        self.setLayout(self.__layout_main)
        self.label_move.raise_()
        self.setWindowTitle(u'YOLOV5-Pytorch')

        '''
        # 设置背景颜色
        palette1 = QPalette()
        palette1.setBrush(self.backgroundRole(),QBrush(QPixmap('background.jpg')))
        self.setPalette(palette1)
        '''

    def slot_init(self):  # 建立通信连接
        self.button_open_camera.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.show_image)
        self.button_close.clicked.connect(self.close)
        self.button_showfps.clicked.connect(self.show_fps)
        # 连接信号和槽
        self.button_showcount.clicked.connect(self.is_show_what_in_pic)
        self.button_save_ori.clicked.connect(self.save_ori)

    # def button_open_camera_click(self):
    #     if self.timer_camera.isActive() == False:
    #         # flag = self.cap.open(self.CAM_NUM)
    #         # if flag == False:
    #         #     msg = QtWidgets.QMessageBox.Warning(self, u'Warning', u'请检测相机与电脑是否连接正确',
    #         #                                         buttons=QtWidgets.QMessageBox.Ok,
    #         #                                         defaultButton=QtWidgets.QMessageBox.Ok)
    #         #     # if msg==QtGui.QMessageBox.Cancel:
    #         #     #                     pass
    #         # else:
    #         self.timer_camera.start(30)
    #         self.button_open_camera.setText(u'关闭相机')
    #     else:
    #         self.timer_camera.stop()
    #         #self.cap.release()
    #         self.label_show_camera.clear()
    #         self.button_open_camera.setText(u'打开相机')

    def button_open_camera_click(self):
        with torch.no_grad():
            self.detect()

    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cancel = QtWidgets.QPushButton()
        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u'关闭', u'是否关闭！')
        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cancel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'确定')
        cancel.setText(u'取消')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            # if self.cap.isOpened():
            #     self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()
        #pipeline.stop()


    def show_fps(self):
        self.weights = 'yolov5m.pt'
    def is_show_what_in_pic(self):
        self.weights = 'yolov5l.pt'
    def save_ori(self):
        self.weights = 'yolov5x.pt'

    def detect(self,save_img=False):
        out, source, weights, view_img, save_txt, imgsz = \
            'inference/output', '0', self.weights, False, False, 640
        self.webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

        # Initialize
        self.device = select_device('0')
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
        half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=self.model.stride.max())  # check img_size
        if half:
            self.model.half()  # to FP16

        # Set Dataloader
        vid_path, vid_writer = None, None
        if self.webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            self.dataset = LoadStreams(source, img_size=imgsz)
        else:
            save_img = True
            self.dataset = LoadImages(source, img_size=imgsz)

        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Run inference
        img = torch.zeros((1, 3, imgsz, imgsz), device=self.device)  # init img
        _ = self.model(img.half() if half else img) if self.device.type != 'cpu' else None  # run once
        self.timer_camera.start(10)

    def show_image(self):
        self.i+=1
        print(self.i)
        t=time.time()
        with torch.no_grad():
            # # Second-stage classifier
            # classify = False
            # if classify:
            #     modelc = load_classifier(name='resnet101', n=2)  # initialize
            #     modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
            #     modelc.to(device).eval()
            #path, img, im0s, vid_cap = next(self.dataset)
            path, img, im0s, img_depth, depth, self.gyro, intrin, vid_cap = next(self.dataset)
            img = torch.from_numpy(img).to(self.device)
            t0 = time.time()
            img = img.half() if self.device.type != 'cpu' else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = self.model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)
            t2 = time_synchronized()

            # Apply Classifier
            # if classify:
            #     pred = apply_classifier(pred, modelc, img, im0s)

            names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if self.webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path('inference/output') / Path(p).name)
                txt_path = str(Path('inference/output') / Path(p).stem) + ('_%g' % self.dataset.frame if self.dataset.mode == 'video' else '')
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    img_ori = im0.copy()
                    # Write results
                    for *xyxy, conf, cls in det:
                    # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                # Print time (inference + NMS)
                # print('%sDone. (%.3fs)' % (s, t2 - t1))
                print('%sDone. (%.3ffps)' % (s, 1/(t2 - t1)))

                # Stream results
                cv2.putText(im0,'%.3ffps' % (1/(t2 - t1)),(0,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
                im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                #img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
                showImage = QtGui.QImage(im0.data, im0.shape[1], im0.shape[0], QtGui.QImage.Format_RGB888)
                self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration


            print('Done. (%.3fs)' % (time.time() - t0))

if __name__ == '__main__':
    App = QApplication(sys.argv)
    win = Ui_MainWindow()
    win.show()
    sys.exit(App.exec_())
