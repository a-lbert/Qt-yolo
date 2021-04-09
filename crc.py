import os
import sys

def calc_crc16(string):
    #data = bytearray.fromhex(string)
    data = string.encode()
    print(data)

    crc = 0xFFFF
    for pos in data:
        crc ^= pos
        print('pos',pos)
        for i in range(8):
            if ((crc & 1) != 0):
                crc >>= 1
                crc ^= 0xA001
            else:
                crc >>= 1


    return ((crc & 0xff) << 8) + (crc >> 8)

class CRC:
    """循环冗余检验

    parameters
    ----------

    info : list
        需要被编码的信息

    crc_n : int, default: 32
        生成多项式的阶数

    p : list
        生成多项式

    q : list
        crc后得到的商

    check_code : list
        crc后得到的余数，即计算得到的校验码

    code : list
        最终的编码

    ----------

    """

    def __init__(self, info, crc_n=32):
        self.info = list(info)
        '''
        输入参数：发送数据比特序列，CRC生成多项式阶数
        '''

        '''
        初始化CRC生成多项式p,其中P和二进制码（多项式比特序列）的关系为如下例：
        G(X)系数:  4 1 0 分别代表x的4次方，x的1次方，x的0次方以此类推
        G(X) =    1*(X^4) + 0*(X^3) + 0*(X^2) + 1*X + 1*1
        二进制码 = 1         0         0         1     1
        可根据需要自行添加修改，也可使用国际标准
        '''

        if crc_n == 8:
            loc = [8, 2, 1, 0]
        elif crc_n == 32:
            loc = [32, 26, 23, 22, 16, 12, 11, 10, 8, 7, 5, 2, 1, 0]  # 国际标准CRC-32
        elif crc_n == 16:
            loc = [16, 15, 2, 0]  # 国际标准CRC-16
        elif crc_n == 4:
            loc = [4, 3, 0]

        # 列表解析转换为多项式比特序列
        p = [0 for i in range(crc_n + 1)]
        for i in loc:
            p[i] = 1
        p = p[::-1]  # 逆序输出

        info = self.info.copy()
        times = len(info)
        n = crc_n + 1

        # 左移补零即乘积
        for i in range(crc_n):
            info.append(0)

        # 乘积除以多项式比特序列
        q = []  # 商
        for i in range(times):
            if info[i] == 1:  # 若乘积位为1，则商1，后逐位异或
                q.append(1)
                for j in range(n):  # n即p的位数
                    info[j + i] = info[j + i] ^ p[j]  # 按位异或
            else:  # 若乘积位是0，则商0，看下一位
                q.append(0)

        # 余数即为CRC编码
        check_code = info[-crc_n::]

        # 生成编码
        code = self.info.copy()
        for i in check_code:
            code.append(i)

        self.crc_n = crc_n
        self.p = p
        self.q = q
        self.check_code = check_code
        self.code = code

    def print_format(self):
        """格式化输出结果"""

        print('{:10}\t{}'.format('发送数据比特序列：', self.info))
        print('{:10}\t{}'.format('生成多项式比特序列：', self.p))
        print('{:15}\t{}'.format('商：', self.q))
        print('{:10}\t{}'.format('余数（即CRC校验码）：', self.check_code))
        # print('{:5}\t{}'.format('带CRC校验码的数据比特序列：', self.code))




if __name__ == "__main__":
    import numpy as np

    m = np.array([1, 1, 0, 0, 1, 1])  # 发送数据比特序列
    m = list(m)  # 转化为列表类型
    crc = CRC(m, 4)
    crc.print_format()