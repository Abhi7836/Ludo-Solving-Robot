import cv2
import numpy as np
import cv2.aruco as aruco
import serial
import time
font=cv2.FONT_HERSHEY_SIMPLEX
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
parameters= aruco.DetectorParameters_create()
#arduino = serial.Serial('COM19',9600)
###############################################################################################################################################
def arucodect(image,m):
        X,Y=int(image.shape[0]/m),int(image.shape[1]/m)
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        mar=np.full((m,m),0)
        corners,ids,rejected=aruco.detectMarkers(image=gray,dictionary=aruco_dict,parameters=parameters)
        if len(corners):
                x=int((corners[0][0][0][0]+corners[0][0][1][0])/2)
                y=int((corners[0][0][0][1]+corners[0][0][2][1])/2)
                j,i=int(x/X),int(y/Y)
                mar[i][j]=1
        return image,ids,mar,x,y,corners
##################################################################################################################################################
def flipimage(mar,image,m):
        if mar[0][(m-1)//2]==1:
                M = cv2.getRotationMatrix2D(center, 90, scale)
                image = cv2.warpAffine(image, M, (h, w))
                i,j=0,(m-1)//2
        elif mar[(m-1)//2][(m-1)]==1:
                M = cv2.getRotationMatrix2D(center, 180, scale)
                image = cv2.warpAffine(image, M, (h, w))
                i,j=(m-1)//2,(m-1)
        elif mar[(m-1)][(m-1)//2]==1:
                M = cv2.getRotationMatrix2D(center, 270, scale)
                image = cv2.warpAffine(image, M, (h, w))
                i,j=(m-1),(m-1)//2
        else:
                i,j=(m-1)//2,0
        return image
############################################################################################################################################################
def shapedetect(masky,maskr,image,m):
    X,Y=int(image.shape[0]/m),int(image.shape[1]/m)
    msy=np.full((m,m),0)
    msr=np.full((m,m),0)
    _, thresholdy = cv2.threshold(masky, 240, 255, cv2.THRESH_BINARY)
    _, contoursy, _ = cv2.findContours(thresholdy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, thresholdr = cv2.threshold(maskr, 240, 255, cv2.THRESH_BINARY)
    _, contoursr, _ = cv2.findContours(thresholdr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contoursy:
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        cv2.drawContours(image, [approx], 0, (0), 1)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        if len(approx) == 3:
            cv2.putText(image, "T", (x, y), font, 1, (0))
            j,i=int(x/X),int(y/Y)
            msy[i][j]=2
        elif len(approx) == 4:
            cv2.putText(image, "S", (x, y), font, 1, (0))
            j,i=int(x/X),int(y/Y)
            msy[i][j]=3
        elif len(approx) == 5:
            cv2.putText(image, "p", (x, y), font, 1, (0))
        else:
            cv2.putText(image, "C", (x, y), font, 1, (0))
            j,i=int(x/X),int(y/Y)
            msy[i][j]=1
    for cnt in contoursr:
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
        cv2.drawContours(image, [approx], 0, (0), 1)
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        if len(approx) == 3:
            cv2.putText(image, "T", (x, y), font, 1, (0))
            j,i=int(x/X),int(y/Y)
            msr[i][j]=2
        elif len(approx) == 4:
            cv2.putText(image, "S", (x, y), font, 1, (0))
            j,i=int(x/X),int(y/Y)
            msr[i][j]=3
        elif len(approx) == 5:
            cv2.putText(image, "p", (x, y), font, 1, (0))
        else:
            cv2.putText(image, "C", (x, y), font, 1, (0))
            j,i=int(x/X),int(y/Y)
            msr[i][j]=1
    return image,msy,msr
############################################################################################################################################################
def colourdetect(image):
    L_yellow = np.array([21,155,220])
    U_yellow= np.array([35,255,255])
    L_red = np.array([0,155,220])
    U_red = np.array([20,255,255])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = np.ones((6,6),np.uint8)
    maskr1 = cv2.inRange(hsv, L_red, U_red)
    masky1 = cv2.inRange(hsv, L_yellow, U_yellow)
    maskr1 = cv2.morphologyEx(maskr1, cv2.MORPH_CLOSE, kernel)
   # maskr1 = cv2.GaussianBlur(maskr1,(15,15),0)
    return maskr1,masky1
############################################################################################################################################################
def matrixr(maskr,m):
    X,Y=int(image.shape[0]/m),int(image.shape[1]/m)
    Mr=np.full((m,m),0)
    for j in range(m):
        for i in range(m):
            a=maskr[int(X/2+j*X),int(Y/2+i*Y)]
            if a==255:
                Mr[j][i]=4
    return Mr
######################################################################################################################################################
def matrixy(masky,m):
    X,Y=int(image.shape[0]/m),int(image.shape[1]/m)
    My=np.full((m,m),0)
    for j in range(m):
        for i in range(m):
            a=masky[int(X/2+j*X),int(Y/2+i*Y)]
            if a==255:
                My[j][i]=1
    return My

#############################################################################################################################
def pathplanning(M,i,j,angle):
        curpos=(i)*9+j
        path=np.array([
          [36,27,18,9,0,1,2,3,4, 5, 6, 7, 8,17,26,35,44,53,62,71,80,79,78,77,76,75,74,73,72,63,54,45,36,37,38,39,40],
          [36,27,18,9,0,1,2,3,4, 5, 6, 7, 8,17,26,35,44,53,62,71,80,79,78,77,76,67,58,57,56,47,38,39,40, 0, 0, 0, 0],
          [36,27,18,9,0,1,2,3,4, 5, 6, 7, 8,17,26,35,44,43,42,51,60,59,58,57,56,47,38,39,40, 0, 0, 0, 0, 0, 0, 0, 0],
          [36,27,18,9,0,1,2,3,4, 5, 6, 7, 8,17,26,35,44,43,42,51,60,59,58,67,76,75,74,73,72,63,54,45,36,37,38,39,40],
          [36,27,18,9,0,1,2,3,4,13,22,23,24,33,42,43,44,53,62,71,80,79,78,77,76,75,74,73,72,63,54,45,36,37,38,39,40],
          [36,27,18,9,0,1,2,3,4,13,22,23,24,33,42,43,44,53,62,71,80,79,78,77,76,67,58,57,56,47,38,39,40, 0, 0, 0, 0],
          [36,27,18,9,0,1,2,3,4,13,22,23,24,33,42,51,60,59,58,57,56,47,38,39,40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [36,27,18,9,0,1,2,3,4,13,22,23,24,33,42,51,60,59,58,67,76,75,74,73,72,63,54,45,36,37,38,39,40, 0, 0, 0, 0]
                    ])
        dist=np.full((81,9),1000)
        dp=np.full((81,9),-1)
        print('Switch on the LED')
        visited=0
        for i in range(0,8):
            visited=0
            for j in range(0,len(path[i])):
                if path[i][j]==39:
                    break
                if path[i][j]==36:
                    if visited==0:
                        visited=1
                    else:
                        continue
                for k in range(j+1,len(path[i])):
                    if path[i][k]==40:
                        break
                    req=path[i][k]
                    if k-j< dist[path[i][j]][M[(req//9)][req%9]]:
                        dist[path[i][j]][M[(req//9)][req%9]]=k-j
                        dp[path[i][j]][M[(req//9)][req%9]]=i  
        home=0
        while home!=1 :
                x=int(input())
                print('switch off the light')
                ste=dist[curpos][x]
                pathans=dp[curpos][x]
                if ste==1000 :
                        print('Switch on the LED')
                        continue
                for i in range(0,37):
                        if path[pathans][i]==curpos:
                                j=i
                                break
                while ste!=0:
                        j+=1
                        d=path[pathans][j]
                        angle=Ardcmd(curpos,d,m,angle)
                        #print(str(angle)+"deg")
                        curpos=path[pathans][j]
                        ste=ste-1
                        #print(curpos,end=' ')
                if curpos==39:
                        curpos=40
                        home=1
                        print("F")
                        print('I won')
                        break
                #print('Iam at',curpos)
                print('Switch on my LED')
def Ardcmd(curpos,d,m,angle):
        Xd,Yd=d//m,d%m
        Xcp,Ycp=curpos//m,curpos%m
        X=Xcp-Xd
        Y=Ycp-Yd
        if angle==0:
                if Ycp-Yd==0:
                        if Xcp-Xd>0:
                                print("F")
                        if Xcp-Xd<0:
                                angle=180
                                print("B")
                if Xcp-Xd==0:
                        if Ycp-Yd<0:
                                angle=90
                                print("RF")
                        if Ycp-Yd>0:
                                angle=270
                                print("L")
        elif angle==90:
                if Ycp-Yd==0:
                        if Xcp-Xd<0:
                                angle=180
                                print("RF")
                        else:
                                angle=0
                                print("L")
                if Xcp-Xd==0:
                        if Ycp-Yd<0:
                                print("F")
                        else:
                                print("B")
                                angle=270
        elif angle==180:
                if Ycp-Yd==0:
                        if Xcp-Xd<0:
                                print("F")
                        else:
                                print("B")
                                angle=0
                if Xcp-Xd==0:
                        if Ycp-Yd>0:
                                angle=270
                                print("RF")
                        else:
                                angle=90
                                print("L")
        elif angle==270:
                if Ycp-Yd==0:
                        if Xcp-Xd>0:
                                angle=0
                                print("RF")
                        else:
                                angle=180
                                print("L")
                if Xcp-Xd==0:
                        if Ycp-Yd>0:
                                print("F")
                        else:
                                print("B")
                                angle=90
        return angle
            
       
########################################################################################################################################################    
if __name__ == "__main__":
                im=cv2.imread("pxm180.png")
                m=9
                angle=0
                showCrosshair = False
                fromCenter = False
                r = cv2.selectROI("Image", im, fromCenter, showCrosshair)
                image = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
                3
                image,ID,mar,x,y,corners=arucodect(image,m)
                (h, w) = image.shape[:2]
                center = (w / 2, h / 2)
                scale = 1.0
                image=flipimage(mar,image,m)
                image,ID,mar,x,y,corners=arucodect(image,m)
                maskr,masky=colourdetect(image)
                mr=matrixr(maskr,m)
                my=matrixy(masky,m)
                image,msy,msr=shapedetect(masky,maskr,image,m)
                M=mr+my+msy+msr
                M[(m-1)//2][0],M[0][(m-1)//2],M[(m-1)//2][(m-1)],M[(m-1)][(m-1)//2],M[(m-1)//2][(m-1)//2]=8,8,8,8,9
                M[(m-1)//2][0]=1
                aruco.drawDetectedMarkers(image,corners)
                cv2.putText(image,str(ID[0][0]), (x,y), font, 1, (0,0,255), 2, cv2.LINE_AA)
                cv2.imshow('img',image)
                print(M)
                print(ID)
                pathplanning(M,4,0,angle)
