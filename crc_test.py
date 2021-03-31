from crc import *
from binascii import *
import binascii

def calc_crc(string):
    #data = bytearray.fromhex(string)
    data = string.encode()
    crc = 0xFFFF
    for pos in data:
        #print(pos)
        crc ^= pos
        for i in range(8):
            if ((crc & 1) != 0):
                crc >>= 1
                crc ^= 0xA001
            else:
                crc >>= 1
    #return hex(((crc & 0xff) << 8) + (crc >> 8))
    return ((crc & 0xff) << 8) + (crc >> 8)

def split_data(i):
    low = hex(i % 256)
    low = "{:#04X}".format(int(low, 16))
    high = hex(int(i / 256))
    high = "{:#04X}".format(int(high, 16))
    return high + low

str = '0X000X000X000X000X000X000X000X87'

crc=binascii.crc32(str.encode()) & 0xffffffff
print('crc',crc)
print(split_data(crc))

crc=calc_crc(str)
print('crc:',crc)



#32:crc:2846520145