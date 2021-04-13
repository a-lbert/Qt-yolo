import sys
import os
import cv2
# git test
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPalette, QBrush, QPixmap
from crcmod import *
from crc import *
from binascii import *
import binascii
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

from SignalSlot import *
import serial
import serial.tools.list_ports
from threading import Thread
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized

# from TuXiangChuLi import cal_angel
from moment import cal_angel
from crc import calc_crc16

def accel_data(accel):
    return np.asarray([accel.x, accel.y, accel.z])


class MainWin(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(MainWin, self).__init__()
        self.setupUi(self)

        self.__flag_work = 0
        self.x = 0
        self.count = 0
        self.fps = 0.0
        self.is_show_fps = False
        self.show_what_in_pic = False
        self.what_in_pic = ''
        self.info = ''
        self.result = ''

        self.gyro = None
        self.serial_bytes = ''
        self.fps = 0
        self.ori_num = 0
        self.pro_num = 0
        self.ori = None
        self.pro = None
        self.dataset = None
        self.weights = 'yolov5s.pt'
        self.i = 0
        self.weights = './3_1.pt'
        self.img_video = None
        self.timer_camera = QtCore.QTimer()  # 初始化定时器

        self.timer_camera.timeout.connect(self.show_image)

        self.timer_ui = QtCore.QTimer()  # 初始化定时器

        self.timer_ui.timeout.connect(self.update_ui)

        # self.timer_camera.timeout.connect(self.receive_data)
        self.serial = serial.Serial()
        self.data_to_send = ''
        self.open_serial_button.clicked.connect(self.open_serial)
        self.close_serial_button.clicked.connect(self.close_serial)
        self.serial_timer = QTimer(self)
        self.serial_timer.timeout.connect(self.receive_data)

        self.is_hex = 1  # 16jinzhi
        # 选择显示控件
        self.label_obj = [self.label_obj1, self.label_obj2, self.label_obj3
                          ]
        self.fire = [self.fire1, self.fire2
                     ]
        self.info_obj = [self.info_obj1, self.info_obj2, self.info_obj3,
                         self.info_obj4, self.info_obj5]
        self.update_intrin = 1
        self.last_time = 0
        self.ppx = 0
        self.ppy = 0
        self.fx = 0
        self.fy = 0
        self.run_thread = 1
        self.open_serial()
        self.detect()


    # 调整图片以显示
    def adapt_img(self, img):
        RGB_pic = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        RGB_pic = QtGui.QImage(RGB_pic.data, RGB_pic.shape[1], RGB_pic.shape[0], QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(RGB_pic)

    def show_pic(self, img, show, Scaled=True):
        RGB_pic = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        RGB_pic = QtGui.QImage(RGB_pic.data, RGB_pic.shape[1], RGB_pic.shape[0], QtGui.QImage.Format_RGB888)
        show.setScaledContents(Scaled)
        show.setPixmap(QtGui.QPixmap.fromImage(RGB_pic))

    # 分割数据，格式化输出
    def split_data(self, i):

        low = hex(i % 256)
        low = "{:#04X}".format(int(low, 16))
        high = hex(int(i / 256))
        high = "{:#04X}".format(int(high, 16))
        return high + low

    def receive_data(self):
        self.info_serial.setText('等待接受数据')
        try:
            num = self.serial.inWaiting()
        except:
            self.close_serial()
            return None
        if num > 0:
            data = self.serial.read(num)
            # print('SHUJU%s'%type(data))
            if self.is_hex == 1:
                out_s = ''
                for i in range(0, len(data)):
                    out_s = out_s + '{:02X}'.format(data[i])
                # for i in range(0, len(out_s)):
                #     print(out_s[i])
                if out_s[:4] == '55AA':
                    if out_s[4:6] == '01':
                        print('01')
                        self.send_data()
                    elif out_s[4:6] == '02':
                        print('02')
                    print('ok')
                print(out_s)
            else:
                for i in range(len(data)):
                    print(data[i])
                self.receive_data_label.setText(str(data.decode('iso-8859-1')))

        pass

    def open_serial(self):
        self.serial.port = str('/dev/ttyTHS0')
        self.serial.baudrate = int(9600)
        self.serial.bytesize = int(8)
        self.serial.stopbits = int(1)
        self.serial.parity = str('N')
        try:
            self.serial.open()
            self.info_serial.setText('串口已经打开')
        except:
            self.info_serial.setText('此串口无法打开')
            return None
        self.serial_timer.start(50)
        if self.serial.isOpen():
            self.open_serial_button.setEnabled(False)
            self.close_serial_button.setEnabled(True)

    def close_serial(self):
        self.serial_timer.stop()
        try:
            self.serial.close()
            self.info_serial.setText('串口已关闭')
            self.send_data_label.clear()
            self.receive_data_label.clear()
        except:
            pass
        self.open_serial_button.setEnabled(True)
        self.close_serial_button.setEnabled(False)

    def send_data(self, data):


        self.send_data_label.setText(str(data))

        if self.serial.isOpen():
            if data != '':
                self.send_data_label.setText(str(data))
                data = (data + '\r\n').encode('utf-8')
                num = self.serial.write(data)
                self.info_serial.setText(str(num))
        self.data_to_send = ''
    def update_ui(self):

        self.img_video = LoadStreams.video(self.dataset)
        self.show_pic(self.img_video[0], self.ShowLabel)
        t = time_synchronized()
        self.fps = 1 / (t - self.last_time)
        self.fps = (self.fps // 2) * 2
        self.info_lab.setText(str(int(self.fps)))
        self.last_time = time_synchronized()


    def update_video(self):

        i = 0

        while True:
            t1 = time_synchronized()
            if i == 0:
                time.sleep(0.5)
            elif i == 1:
                time.sleep(0.03)
                # self.show_pic(img_video[0], self.ShowLabel)

            self.img_video = LoadStreams.video(self.dataset)

            i = 1
            t2 = time_synchronized()
            self.fps = 1 / (t2 - t1)
            self.fps = (self.fps // 2) * 2
            self.info_lab.setText(str(int(self.fps)))
            # print('子线程显示图像')

    def show_image(self):
        self.i += 1
        # Save_path = './inference/output'
        S_path = '/home/limeng/Qt-yolo/data_to-cal'
        print('当前获取第{}帧'.format(self.i))
        t = time.time()
        rgb_path = '../exp/c/' + 'rgb' + str(self.i) + '.jpg'
        dep_path = '../exp/c/' + 'dep' + str(self.i) + '.jpg'
        dep_txt_path = '../exp/c/' + 'dep' + str(self.i) + '.txt'
        # print(rgb_path,dep_path)
        with torch.no_grad():
            path, img, im0s, img_depth, depth, self.gyro, intrin, vid_cap = next(self.dataset)
            # test_depth = np.asanyarray(depth.get_data())
            # print('type:', type(test_depth), type(depth),test_depth.shape)


            # depth.get_data(): #<class 'pyrealsense2.pyrealsense2.BufData'>
            #print('type:',type(img_depth),type(depth))
            # type: # <class 'list'>    <class 'pyrealsense2.pyrealsense2.depth_frame'>
            # cv2.imwrite('./test.jpg', img)

            if self.update_intrin == 1:
                self.ppx = intrin[0]
                self.ppy = intrin[1]
                self.fx = intrin[2]
                self.fy = intrin[3]
                self.update_intrin = 0

                print('更新内参')
                # cv2.imwrite('./test.jpg', img)

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

            names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if self.webcam:  # batch_size >= 1
                    p, s, im0, img_depth = path[i], '%g: ' % i, im0s[i].copy(), img_depth[i].copy()
                else:
                    p, s, im0, img_depth = path, '', im0s, img_depth

                save_path = str(Path('inference/output') / Path(p).name)
                txt_path = str(Path('inference/output') / Path(p).stem) + (
                    '_%g' % self.dataset.frame if self.dataset.mode == 'video' else '')

                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(img_depth, alpha=0.03), cv2.COLORMAP_JET)
                # print('type:', type(depth_colormap),depth_colormap.shape)
                # type: #<class 'numpy.ndarray'> (720, 1280, 3)
                # print('type:', type(im0s[0]), im0s[0].shape)
                # type: #<class 'numpy.ndarray'> (720, 1280, 3)
                self.show_pic(depth_colormap, self.depth_label)

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    self.info = ''
                    # xuanzekongjian
                    obj_i = 0;
                    fire_i = 0;
                    #cv2.imwrite(save_path, img)
                    test_depth = np.asanyarray(depth.get_data())
                    for *xyxy, conf, cls in det:
                        # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        _c1 = (int(xyxy[1]) // 8 + 1) * 8
                        _c3 = (int(xyxy[3]) // 8 + 1) * 8
                        _c0 = (int(xyxy[0]) // 8 + 1) * 8
                        _c2 = (int(xyxy[2]) // 8 + 1) * 8
                        obj_1 = im0s[0][_c1:_c3, _c0:_c2]
                        obi_depth = img_depth[_c1:_c3, _c0:_c2]
                        test_depth = test_depth[_c1:_c3, _c0:_c2]

                        # if self.i%5 ==0:
                        #     cv2.imwrite(rgb_path,obj_1)
                        #     cv2.imwrite(dep_path,obi_depth)
                        #     np.savetxt(dep_txt_path, test_depth, fmt="%d", delimiter=",")
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        # self.info += str(round(depth.get_distance(int(xyxy[1].item()),int(xyxy[2].item())), 6)) + '\n'
                        pixel_x, pixel_y = int((xyxy[0].item() + xyxy[2].item()) / 2), int(
                            (xyxy[1].item() + xyxy[3]) / 2)
                        # z为深度，x指向双目相机，y向下，右手坐标系
                        z = depth.get_distance(pixel_x, pixel_y)
                        x, y = [(pixel_x - self.ppx) * z / self.fx, (pixel_y - self.ppy) * z / self.fy]
                        self.data_to_send += self.split_data(int(1000 * x))
                        self.data_to_send += self.split_data(int(1000 * y))
                        self.data_to_send += self.split_data(int(1000 * z))
                        print('识别出目标：{} 像素坐标：（{},{}）实际坐标（mm）：({:.3f},{:.3f},{:.3f})'.format(
                            names[int(cls)], pixel_x, pixel_y, x * 1000, y * 1000, z * 1000
                        ))
                        self.info += names[int(cls)] + ':'
                        self.result = ''
                        self.result += names[int(cls)] + ':'
                        self.result += str('({:.0f},{:.0f},{:.0f})'.format(
                            x * 1000, y * 1000, z * 1000
                        )) + '\n'
                        if names[int(cls)] == 'pipe':
                            print('计算角度')
                            theta, width = cal_angel(obj_1)
                            width = width * z * 1000 / self.fx
                            if (obj_i > 3):
                                obj_i = obj_i % 3
                            self.show_pic(obj_1, self.label_obj[obj_i], True)
                            self.data_to_send += self.split_data(int(width))
                            self.data_to_send += self.split_data(int(theta))
                            # 表示物体种类，留做接口
                            self.data_to_send += '0X00'
                            crc = calc_crc16(self.data_to_send)


                            self.data_to_send += self.split_data(crc)

                            self.result += 'angel:'
                            self.result += str(int(theta))
                            self.result += 'width:'
                            self.result += str(int(width))
                            self.info_obj[obj_i].setText(self.result)
                            obj_i += 1
                        elif names[int(cls)] == 'fire':
                            self.data_to_send += '0X01'

                            crc = calc_crc16(self.data_to_send)
                            self.data_to_send += self.split_data(crc)
                            self.send_data(self.data_to_send)
                            self.data_to_send = ''
                            if (fire_i > 2):
                                fire_i = fire_i % 2
                            self.show_pic(obj_1, self.fire[fire_i], True)
                            fire_i += 1


                print('处理完成，当前帧率(%.3ffps)' % self.fps)
                self.show_pic(im0, self.yolo_label)

                self.imu_label.setText(str(self.gyro))
                self.info_result.setText(self.result)

                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

    def detect(self, save_img=False):
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
        self.timer_camera.start(50)
        self.timer_ui.start(30)
        # up_thread = Thread(target=self.update_video, args=(), daemon=True)
        # up_thread.start()

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

            event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    mainWindow = MainWin()

    mainWindow.show()
    sys.exit(app.exec_())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# import PySide2
# import os
# dirname = os.path.dirname(PySide2.__file__)
# plugin_path = os.path.join(dirname, 'plugins', 'platforms')
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
# from PySide2 import QtWidgets,QtCore
