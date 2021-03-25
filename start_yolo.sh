#!/bin/bash
#sleep 30
echo ' ' | sudo -S  chown sz2 /dev/ttyTHS0
cd /home/sz2/work/Qt-yolo
python3 main.py
#/usr/bin/python3.6 /home/sz2/work/Qt-yolo/main.py
#./pycharm.sh
exit 0
#.bashrc
#gnome-terminal --geometry=80x25+10+10 -- bash -c "/home/sz2/start_yolo.sh;exec #bash;"