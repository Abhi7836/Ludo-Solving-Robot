from __future__ import print_function
import pyzbar.pyzbar as pyzbar
import cv2
import numpy as np
import imutils

lower = {'red':(152, 58, 180), 'pink':(124, 68, 131), 'blue':(100, 147, 105), 'yellow':(21, 58, 161)}
upper = {'red':(255,176,255), 'pink':(161,168,255), 'blue':(117,255,255), 'yellow':(33,154,255)}
colors = {'red':(0,0,255), 'pink':(0,255,0), 'blue':(255,0,0), 'yellow':(0, 255, 217), 'orange':(0,140,255)}
fontface=cv2.FONT_HERSHEY_SIMPLEX
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    frame=imutils.resize(frame,width=600)
    blurred=cv2.GaussianBlur(frame,(11,11),0)
    decodedObjects = pyzbar.decode(frame)
    hsv=cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)
    for key, value in upper.items():
        kernel = np.ones((9,9),np.uint8)
        mask = cv2.inRange(hsv, lower[key], upper[key])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        cv2.drawContours(frame,cnts,-1,colors[key],1)
    if(len(cnts)!=0):
       x,y,w,h=cv2.boundingRect(cnts[0])
       cv2.rectangle(frame,(x,y),(x+w,y+h),colors[key],1)
       cv2.putText(frame,key,(x,y),fontface,2,(0,0,255),2);
    for decodedObject in decodedObjects:
        ver = decodedObject.polygon
        n=len(ver)
        for j in range(0,n):
            cv2.line(frame, ver[j], ver[ (j+1) % n], (0,0,255), 3)
    for obj in decodedObjects:
        cv2.putText(frame,str(obj.data),(170,110), fontface, 1, (0,0,200), 2, )
       
    cv2.imshow('frame',frame)

    k=cv2.waitKey(5)&0xFF
    if k==27:
       break
cv2.destroyAllWindows()
cap.release()
    
