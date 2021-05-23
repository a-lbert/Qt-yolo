import socket
import cv2
import numpy
from time import sleep
import time
from threading import Thread

# # socket.AF_INET 用于服务器与服务器之间的网络通信
# # socket.SOCK_STREAM 代表基于TCP的流式socket通信
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# # 连接服务端
# address_server = ('192.168.0.109', 8888)
# sock.connect(address_server)
# # 从摄像头采集图像
# # 参数是0,表示打开笔记本的内置摄像头,参数是视频文件路径则打开视频
# capture = cv2.VideoCapture(0)
# # capture.read() 按帧读取视频
# # ret,frame 是capture.read()方法的返回值
# # 其中ret是布尔值，如果读取帧正确，返回True;如果文件读到末尾，返回False。
# # frame 就是每一帧图像，是个三维矩阵
# ret, frame = capture.read()
# encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
# while ret:
#     # 首先对图片进行编码，因为socket不支持直接发送图片
#     # '.jpg'表示把当前图片frame按照jpg格式编码
#     result, img_encode = cv2.imencode('.jpg', frame)
#     img_encode = cv2.imencode('.jpg', frame, encode_param)[1]
#     data = numpy.array(img_encode)
#     stringData = data.tobytes()
#     print(stringData)
#     print((str(len(stringData)).ljust(16)).encode())
#     # 首先发送图片编码后的长度
#     sock.send((str(len(stringData)).ljust(16)).encode('utf-8'))
#     # 然后一个字节一个字节发送编码的内容
#     # 如果是python对python那么可以一次性发送，如果发给c++的server则必须分开发因为编码里面有字符串结束标志位，c++会截断
#     # for i in range(0, len(stringData)):
#     #     sock.send(stringData[i])
#     sock.send(stringData)
#     sleep(1)
#     ret, frame = capture.read()
#     cv2.resize(frame, (640, 480))
#
# sock.close()
# cv2.destroyAllWindows()
#
#
#
# 收
class udp(Thread):
    def __init__(self, func, args, name='udp'):
        Thread.__init__(self)
        self.name = name
        self.func = func
        self.args = args
    def run(self):
        self.func(*self.args)

def recv_size(sock, count):
    buf = bytes()
    while count:
        newbuf = sock.recv(count)
        #print(newbuf)
        if not newbuf: return None
        #print(newbuf.decode('utf-8',errors='ignore'))
        buf+=newbuf
        #buf += newbuf.decode('utf-8',errors='ignore')
        #print(newbuf)
        #print(buf.encode())
        count -= len(newbuf)
        # print(count)
    return buf
# socket.AF_INET 用于服务器与服务器之间的网络通信
#socket.SOCK_STREAM 代表基于TCP的流式socket通信
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 设置地址与端口，如果是接收任意ip对本服务器的连接，地址栏可空，但端口必须设置
address = ('', 8888)
s.bind(address)  # 将Socket（套接字）绑定到地址
s.listen(True)   # 开始监听TCP传入连接
print ('Waiting for images...')

# 接受TCP链接并返回（conn, addr），其中conn是新的套接字对象，可以用来接收和发送数据，addr是链接客户端的地址。

conn, addr = s.accept()
while True:
    t1=time.time()
    length = recv_size(conn,16)  # 首先接收来自客户端发送的大小信息
    #print(length)
    if isinstance(length.decode(), str):   # 若成功接收到大小信息，进一步再接收整张图片
        stringData = recv_size(conn,int(length))
        #print(stringData)
        data =numpy.frombuffer(stringData, dtype='uint8')
        decimg = cv2.imdecode(data,1)  # 解码处理，返回mat图片
        #print(decimg)
        cv2.imshow('SERVER', decimg)
        if cv2.waitKey(1) == 27:
            break
        print('Image recieved successfully!')
        if cv2.waitKey(1) == 27:
            break
    t2=time.time()
    print(t2-t1)
s.close()
cv2.destroyAllWindows()