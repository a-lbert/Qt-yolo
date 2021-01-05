import sys
import os
import cv2
#git test
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QPalette, QBrush, QPixmap
from crcmod import *
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

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized


def accel_data(accel):
    return np.asarray([accel.x, accel.y, accel.z])


class MainWin(QMainWindow,Ui_MainWindow):

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
        self.info=''
        self.result=''

        self.gyro=None
        self.serial_bytes=''
        self.fps=0
        self.ori_num = 0
        self.pro_num = 0
        self.ori = None
        self.pro = None
        self.dataset = None
        self.weights = 'yolov5s.pt'
        self.i = 0
        self.weights = 'best14.pt'
        self.timer_camera = QtCore.QTimer()  # 初始化定时器
        self.timer_camera.timeout.connect(self.show_image)
        #self.timer_camera.timeout.connect(self.receive_data)
        self.serial = serial.Serial()
        self.data_to_send = ''
        self.open_serial_button.clicked.connect(self.open_serial)
        self.close_serial_button.clicked.connect(self.close_serial)
        self.serial_timer = QTimer(self)
        self.serial_timer.timeout.connect(self.receive_data)
        self.is_hex = 1 #16jinzhi

        self.detect()
        self.open_serial()

        #self.show_image()
        # self.yolo_thread=YoloThread()
        # self.yolo_thread.emit_pic.connect(self.update_ui)
        # self.yolo_thread.start()
        # self.yolo_thread.exec()
    #调整图片以显示
    def adapt_img(self,img):
        RGB_pic = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        RGB_pic = QtGui.QImage(RGB_pic.data, RGB_pic.shape[1], RGB_pic.shape[0], QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(RGB_pic)
    def show_pic(self,img,show):
        RGB_pic = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        RGB_pic = QtGui.QImage(RGB_pic.data, RGB_pic.shape[1], RGB_pic.shape[0], QtGui.QImage.Format_RGB888)
        show.setScaledContents(True)
        show.setPixmap(QtGui.QPixmap.fromImage(RGB_pic))




    def receive_data(self):
        self.info_serial.setText('等待接受数据')
        try:
            num=self.serial.inWaiting()
        except:
            self.close_serial()
            return None
        if num > 0:
            data = self.serial.read(num)
            #print('SHUJU%s'%type(data))
            if self.is_hex==1:
                out_s=''
                for i in range(0,len(data)):
                    out_s=out_s + '{:02X}'.format(data[i])
                # for i in range(0, len(out_s)):
                #     print(out_s[i])
                if out_s[:4]=='55AA':
                    if out_s[4:6]=='01':
                        print('01')
                        self.send_data()
                    elif out_s[4:6]=='02':
                        print('02')
                    print('ok')
                print(out_s)
            else:
                for i in range(len(data)):
                    print(data[i])
                self.receive_data_label.setText(str(data.decode('iso-8859-1')))

        pass
    def open_serial(self):
        self.serial.port=str('/dev/ttyTHS0')
        self.serial.baudrate=int(9600)
        self.serial.bytesize=int(8)
        self.serial.stopbits=int(1)
        self.serial.parity=str('N')
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
    def send_data(self):
        print('检测结果')
        # print(self.info)
        print(self.data_to_send)
        print(type(self.data_to_send))
        self.send_data_label.setText(str(self.data_to_send))
        #self.send_data_label.setText(str(self.data_to_send))
        #self.imu_label.setText(str(self.gyro))
        #self.info_lab.setText(str(int(self.fps)))
        if self.serial.isOpen():
            #self.info_serial.setText(self.data_to_send)

            if self.data_to_send!='':
                self.send_data_label.setText(str(self.data_to_send))
                data=(self.data_to_send+'\r\n').encode('utf-8')
                num=self.serial.write(data)
                self.info_serial.setText(str(num))
        #self.data_to_send = ''

    def show_image(self):
        self.i+=1
        print(self.i)
        t=time.time()
        with torch.no_grad():
            path, img, im0s, img_depth,depth,self.gyro,intrin,vid_cap = next(self.dataset)

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
                    p, s, im0 ,img_depth= path[i], '%g: ' % i, im0s[i].copy(),img_depth[i].copy()
                else:
                    p, s, im0 ,img_depth= path, '', im0s,img_depth

                save_path = str(Path('inference/output') / Path(p).name)
                txt_path = str(Path('inference/output') / Path(p).stem) + ('_%g' % self.dataset.frame if self.dataset.mode == 'video' else '')
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                #显示原图
                self.show_pic(im0,self.ShowLabel)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(img_depth, alpha=0.03), cv2.COLORMAP_JET)
                self.show_pic(depth_colormap,self.depth_label)


                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    self.info=''
                    self.result = ''
                    for *xyxy, conf, cls in det:
                    # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        self.result += names[int(cls)]+':'
                        print('xyxy',str(xyxy),len(xyxy))

                        self.info += names[int(cls)] + ':'
                        #self.info += str(round(depth.get_distance(int(xyxy[1].item()),int(xyxy[2].item())), 6)) + '\n'
                        x,y=int((xyxy[0].item()+xyxy[2].item())/2),int((xyxy[1].item()+xyxy[3])/2)
                        z=depth.get_distance(x,y)
                        self.result+=str('%.3f'%z)+'\n'

                        self.info_result.setText(self.result)

                        # depth_point=rs.rs2_deproject_pixel_to_point(intrin,[x,y],z)
                        # depth_point1 = rs.rs2_deproject_pixel_to_point(intrin, [x+10, y + 10], z)
                        # depth_point2 = rs.rs2_deproject_pixel_to_point(intrin, [x+10, y - 10], z)
                        #
                        # p1 = np.array(depth_point) - np.array(depth_point1)
                        # p2 = np.array(depth_point) - np.array(depth_point2)
                        # c = np.cross(p1, p2)
                        # c /= np.linalg.norm(c)
                        # d = np.array([0, 0, 1])
                        # coscd = -c.dot(d) / (np.linalg.norm(c) * np.linalg.norm(d))
                        # print(np.arccos(coscd) * 180 / np.pi)
                        # print('depth_point:%s'%str(depth_point))
                        # print('depth_point:%s' % str(depth_point1))
                        # print('depth_point:%s' % str(depth_point2))
                        #
                        #
                        # self.info += str(z)+'米'+'\n'
                        # self.data_to_send = ''
                        # li=[3.2,4.3,z,49,99,5]
                        # for i in li:
                        #     i=int(i*100)
                        #     low=hex(i%256)
                        #     low= "{:02X}".format(int(low,16))
                        #     high=hex(int(i/256))
                        #     high= "{:02X}".format(int(high,16))
                        #     self.data_to_send +=low
                        #     self.data_to_send += high
                        # for i in range(len(self.data_to_send)):
                        #     print(self.data_to_send[i])
                        #
                        # #self.send_data()
                        # crc=binascii.crc32(binascii.a2b_hex(self.data_to_send)) & 0xffffffff
                        # print('crc:%d'%crc)
                        # #self.serial_bytes=bytes(self.data_to_send)
                        # #print(self.data_to_send)

                self.fps=1/(t2 - t1)
                print('%sDone. (%.3ffps)' % (s, self.fps))
                #显示处理结果
                #cv2.putText(im0,'%.3ffps' % (1/(t2 - t1)),(0,25),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
                self.show_pic(im0,self.yolo_label)
                self.info_lab.setText(str(int(self.fps)))
                self.imu_label.setText(str(self.gyro))

                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            print('Done. (%.3fs)' % (time.time() - t0))
    def detect(self,save_img=False):
        out, source, weights, view_img, save_txt, imgsz = \
            'inference/output', '0',self.weights, False, False, 640
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
        self.timer_camera.start(30)

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
            event.accept()

if __name__=='__main__':
    app=QApplication(sys.argv)
    mainWindow=MainWin()
    mainWindow.show()
    sys.exit(app.exec_())


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# import PySide2
# import os
# dirname = os.path.dirname(PySide2.__file__)
# plugin_path = os.path.join(dirname, 'plugins', 'platforms')
# os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
# from PySide2 import QtWidgets,QtCore