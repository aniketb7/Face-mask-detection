from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from openpyxl import Workbook , load_workbook
import time
from datetime import datetime as dt
import numpy as np
import time
import cv2

def detect_and_predict_mask(frame, faceNet, maskNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))
	faceNet.setInput(blob)
	detections = faceNet.forward()
	faces = []
	locs = []
	preds = []
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > 0.5:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			faces.append(face)
			locs.append((startX, startY, endX, endY))
	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)
	return (locs, preds)

print("[INFO] loading face detector model...")
prototxtPath = "D:/ENTC/Face Mask Detection/face-detector/face_detector/deploy.prototxt"
weightsPath = "D:/ENTC/Face Mask Detection/face-detector/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
print("[INFO] loading face mask detector model...")
maskNet = load_model("D:/ENTC/Face Mask Detection/face-detector/mask_detector.model")
print("[INFO] starting video stream...")
cap = cv2.VideoCapture(0)
time.sleep(2.0)
while True:
        check,frame=cap.read()
        frame = cv2.resize(frame, (400,400),interpolation = cv2.INTER_CUBIC)
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        for (box, pred) in zip(locs, preds):
	        (startX, startY, endX, endY) = box
	        (AK, BG, LP, MZ, SB, SP) = pred
	        label='Abdul Kalam'
	        cv2.putText(frame,"{}: {:.2f}%".format(label, max(AK, BG, LP, MZ, SB, SP) * 100), (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,255,0), 2)
	        cv2.rectangle(frame, (startX, startY), (endX, endY),(0,255,0), 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("y"):
	        break


cap.release()
cv2.destroyAllWindows()
wk=load_workbook(filename="D:/ENTC/Face Mask Detection/attendance.xlsx")
sp = wk.active
now=dt.now()
date=now.strftime("%Y-%m-%d")
sp['B1'] = date
time=now.strftime("%H:%M:%S")
for i in range(1,8):
    if(sp['A'+str(i)].value == label):
        x=i
        break

for j in range(ord('A'),ord('D')+1):
    if(sp[chr(j) +'1'].value == date):
      sp[chr(j) + str(x)] = time
      break
wk.save("D:/ENTC/Face Mask Detection/attendance.xlsx")
wk.close()
print("Attendance Recorded")
     

