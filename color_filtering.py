import cv2
import numpy as np


cap=cv2.VideoCapture(0)
kernelOpen=np.ones((5,5))
kernelClose=np.ones((20,20))
lower=np.array([0,72,168])
upper=np.array([12,196,255])

while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(340,220))
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
   
    mask=cv2.inRange(hsv,lower,upper)
    maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
    maskFinal=maskClose
    im,conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)    
    cv2.drawContours(frame,conts,-1,(255,0,0),1)
    if(len(conts)!=0):
       x,y,w,h=cv2.boundingRect(conts[0])
       cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
       cxp=int(x+w/2)
       cyp=int(y+h/2)
       cx=cxp-170
       if(cx>15):
          print('left')
       elif(cx<-15):
          print('right')
       elif(cx>-15&cx<15):
           print('stop')
       
    cv2.imshow('frame',frame)

    k=cv2.waitKey(5)&0xFF
    if k==27:
       break
cv2.destroyAllWindows()
cap.release()
    
