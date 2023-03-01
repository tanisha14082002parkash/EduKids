import os
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
# import js2py
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow

def preprocess_image(image):
    # image= cv2.resize(image, (28, 28))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Thresholding
    cv2.imshow("gray",gray)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow("san",thresh)
    thresh= cv2.resize(thresh, (28, 28))
    imgg=thresh.reshape(1,28,28,1)
    # img=thresh
    # cv2.imshow("san",img)
    imgg=imgg.astype('float32')
    imgg=imgg/255
    return imgg
def letter_recognition():
    pred_digit=1
    tipid=[4,8,12,16,20]
    folderpath = os.getcwd()+'\\static\\header'
    list=os.listdir(folderpath)
    overlay=[]
    for i in list:
        image=cv2.imread(f'{folderpath}/{i}')
        overlay.append(image)
    header=overlay[0]
    brush=20
    drawcolor=(0,0,255)
    xp,yp=0,0
    model = load_model('best_model.h5')
    cap=cv2.VideoCapture(0)
    # cap=cv2.VideoCapture(0)
    paintWindow = np.zeros((480,640,3),np.uint8)+255
    paint = np.zeros((480,640,3),np.uint8)
    mphands = mp.solutions.hands
    hands=mphands.Hands(max_num_hands=1, min_detection_confidence=0.2)
    mpDraw=mp.solutions.drawing_utils
    while True:
        success,img=cap.read()
        if success:
            img=cv2.flip(img,1)
            img = cv2.rectangle(img, (200,150), (450,400), (0,255,255), 2)
            results=hands.process(img)
            lmlist=[]
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    for id,lm in enumerate(handLms.landmark):
                        h,w,c=img.shape
                        cx,cy=int(lm.x*w) ,int(lm.y*h)
                        # print(cx,cy)
                        lmlist.append([id,cx,cy])
                    mpDraw.draw_landmarks(img,handLms,mphands.HAND_CONNECTIONS)
                if len(lmlist) !=0:
                    # xp,yp=0,0
                    x1,y1=lmlist[8][1:]
                    x2,y2=lmlist[12][1:]
                    finger=[]
                            # thumb
                    if lmlist[tipid[0]][1]< lmlist[tipid[0]-1][1]:
                        finger.append(1)
                    else:
                        finger.append(0) 
                                #fingers  
                    for id in range(1,5):
                        if lmlist[tipid[id]][2]< lmlist[tipid[id]-2][2]:
                            finger.append(1)
                        else:
                            finger.append(0)    
                    if finger[1] and finger[2]:
                        xp,yp=0,0
                        # print("selection mode")
                        if  0<= y1 <= 100:
                            if 50<x1<120:
                                header=overlay[3]
                                drawcolor=(255,255,255)
                            elif 135<x1<240:
                                header=overlay[0]   
                                drawcolor=(0,0,255)
                            elif 270<x1<375:
                                header=overlay[1]
                                drawcolor=(0,255,255)
                            elif 395<x1<510:
                                header=overlay[2]
                                drawcolor=(255,0,0)
                            elif 520<x1<635:
                                header=overlay[4]
                                xx = paintWindow[150:400,200:450]
                                cv2.imshow("paint",xx)
                                imgg = cv2.cvtColor(xx.astype(np.uint8), cv2.COLOR_BGR2RGB)
                                # image = cv2.cvtColor(paintWindow, cv2.COLOR_)
                                # print(frame.shape)
                                # cv2.imshow("before",img)
                                # print(img.shape)
                                # imgg= cv2.resize(imgg, (28, 28))
                                # img.save("save.jpg")
                                # cv2.imshow("resize",img)
                                # print(img.shape)
                                image=preprocess_image(imgg)
                            #     # image.reshape()
                                # cv2.imshow("san",image)
                            #     # Make prediction
                                prediction = model.predict(image)
                                # print(prediction)
                                digit = np.argmax(prediction)

                            #     # Print the predicted digit
                                # print('Predicted digit:', digit)
                                if pred_digit==digit:
                                    # what to do here
                                    print("yes")
                                    # result = {'status': 'success', 'image': 'new_image.jpg'}
  
       

  

                                else:
                                    print("hi")
                                    # result = {'status': 'failure'}
                                # return jsonify(result)
    
                               


                        cv2.rectangle(img,(x1,y1-5),(x2,y2+5),drawcolor,cv2.FILLED)    
                            
                        
                            
                    elif(finger[1] and finger[2]==False):
                        # fore_finger = (lmlist[8][0],lmlist[8][1])
                        # center = fore_finger
                        if   155<=y1<=395 and 210<=x1<=445:
                            if xp==0 and yp==0:
                                xp,yp=x1,y1
                            cv2.circle(img, (x1,y1), 10, drawcolor,cv2.FILLED)
                            if drawcolor==(255,255,255):
                                brush=50
                                cv2.line(img,(xp,yp),(x1,y1),drawcolor,brush)
                                cv2.line(paintWindow,(xp,yp),(x1,y1),drawcolor,brush)
                                cv2.line(paint,(xp,yp),(x1,y1),drawcolor,brush)
                            else:
                                cv2.line(img,(xp,yp),(x1,y1),drawcolor,30)
                                cv2.line(paintWindow,(xp,yp),(x1,y1),drawcolor,30)
                                cv2.line(paint,(xp,yp),(x1,y1),drawcolor,30)
                            xp,yp=x1,y1 
                        else:
                            xp,yp=0,0
                    else:
                        xp,yp=0,0
            else:
                xp,yp=0,0               
                
            gray=cv2.cvtColor(paint,cv2.COLOR_BGR2GRAY)
            # cv2.imshow("dd",gray)
            # cv2.imshow("can",gray)
            _,inv=cv2.threshold(gray,50,255,cv2.THRESH_BINARY_INV)
            # inv=cv2.cvtColor(inv,cv2.COLOR_BGR2GRAY)
            # cv2.imshow("can",inv)
            inv=cv2.cvtColor(inv,cv2.COLOR_GRAY2BGR)
            # cv2.imshow("can",inv)
            img=cv2.bitwise_and(img,inv)
            # _,invv=cv2.threshold(paintWindow,50,255,cv2.THRESH_BINARY_INV)
            img=cv2.bitwise_or(img,paint)
            header= cv2.resize(header, (640, 100))
            img[0:100,0:640]=header
            # img=cv2.addWeighted(img,0.2,paintWindow,0.2,0)    

            # imggray=cv2.cvtColor(paintWindow,cv2.COLO)

                    # cv2.circle(img,(x1,y1),15,(0,255,0),cv2.FILLED)
                    # print("drawing mode")

        # cv2.imshow("Paint", paintWindow)

            if cv2.waitKey(1) == ord('q'):
                break
            # ret,jpeg=cv2.imencode('.jpg',img)
            # img=jpeg.tobytes()
            # frame=img
            # yield (b'--frame\r\n'
            #    b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
            
            cv2.imshow("img",img)
            # cv2.imshow("san",paintWindow)
            
            # cv2.waitKey(1)
        else:
            break
    # cap.release()
    # cv2.destroyAllWindows()
    cap.release()
    cv2.destroyAllWindows()
    # return -1;       
if __name__=='__main__':
    letter_recognition()
    
