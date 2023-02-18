import time as delay
from datetime import datetime as dt
from openpyxl import Workbook , load_workbook

wk=load_workbook(filename="D:/ENTC/Face Mask Detection/attendance.xlsx")
sp = wk.active
now=dt.now()
date=now.strftime("%Y-%m-%d")
for j in range(ord('A'),ord('Z')+1):
    if(sp[chr(j) +'1'].value == None):
      sp[chr(j) + '1'] = date 
      break
wk.save("D:/ENTC/Face Mask Detection/attendance.xlsx")
wk.close()

while True:
    stream0=open("D:/ENTC/Face Mask Detection/face-mask-detector/detect_mask_video.py")
    read0 = stream0.read()
    exec(read0)
    stream1=open("D:/ENTC/Face Mask Detection/face-detector/detect_face_video.py")
    read1 = stream1.read()
    exec(read1)
    delay.sleep(30)
    if(key == ord("Q")):
        break


