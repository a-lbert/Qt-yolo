#!/bin/bash
#sleep 30
echo ' ' | sudo -S  chown sz2 /dev/ttyTHS0
cd /home/sz2/work/Qt-yolo
python3 main.py
#/usr/bin/python3.6 /home/sz2/work/Qt-yolo/main.py
#./pycharm.sh
exit 0
