from flask import Flask,render_template,redirect,request,session,url_for,message_flashed,flash,Response
import smtplib
import json
import email.utils
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from random import randint
from http.client import HTTPResponse
import cx_Oracle
import hashlib
import os
from datetime import timedelta
# from hand_counter import hand_tracker
# from virtual_painter import letter_recognition
import cv2
import os
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import tensorflow
import time
# import * as bcrypt from 'bcrypt'
conn=cx_Oracle.connect('system/agarwal@//localhost:1521/orcl')
cur=conn.cursor()
# cur.execute("CREATE TABLE  IF NOT EXISTS EDUKIDS (USERNAME VARCHAR2(20) PRIMARY KEY,FIRST VARCHAR2(20),SECOND VARCHAR(20),PASSWORD VARCHAR2(20),AGE NUMBER(2),GENDER VARCHAR2(20), EMAIL VARCHAR2(23))")
# conn.commit()

app = Flask(__name__)
app.secret_key="super secret keyy"
# app.permanent_session_lifetime=timedelta(days=365)

def preprocess_image(image):
    # image= cv2.resize(image, (28, 28))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Thresholding
    # cv2.imshow("hi",gray)
    # ret, jpeg = cv2.imencode('.jpg', gray)
    # yield (b'--frame\r\n'
    #            b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    rett, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # cv2.imshow("san",thresh)
    # cv2.imshow("hi",thresh)

    # time.sleep(5)
    thresh= cv2.resize(thresh, (28, 28))
    
    imgg=thresh.reshape(1,28,28,1)
    # img=thresh
    # cv2.imshow("san",img)
    imgg=imgg.astype('float32')
    imgg=imgg/255
    return imgg

# @app.route('/letter_recognition')
def letter_recognition(pred_digit):
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
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH,480)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,600)
    # cap=cv2.VideoCapture(0)
    paintWindow = np.zeros((480,640,3),np.uint8)+255
    paint = np.zeros((480,640,3),np.uint8)
    mphands = mp.solutions.hands
    hands=mphands.Hands(max_num_hands=1, min_detection_confidence=0.5)
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
                # cv2.imshow("hiiiiiiiiii",img)
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
                                drawcolor=(0,0,0)
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
                                # cv2.imshow("san",xx)
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
                                    
                                    digitImagePath=os.getcwd()+'//static//hurahh.jfif'
                                    pred_digit=(pred_digit+1)%10
                                    digitImage=cv2.imread(digitImagePath)
                                    mergedImage = mergeImage(img,digitImage)
                                    ret, jpeg = cv2.imencode('.jpg', mergedImage)
                                    
                                    for i in range(1,1000):

                                        yield (b'--frame\r\n'
                                            
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                                    # cv2.waitKey(1)
                                    # time.sleep(5)
                                    print("yes")
                                    paintWindow = np.zeros((480,640,3),np.uint8)+255
                                    paint = np.zeros((480,640,3),np.uint8)
                                    # result = {'status': 'success', 'image': pred_digit}
                                    # return Response(letter_recognition(pred_digit),mimetype='multipart/x-mixed-replace; boundary=frame')
                                else:
                                    print(digit)
                                    digitImagePath=os.getcwd()+'//static//try_again.jpg'
                                    
                                    digitImage=cv2.imread(digitImagePath)
                                    mergedImage = mergeImage(img,digitImage)
                                    ret, jpeg = cv2.imencode('.jpg', mergedImage)
                                    # cv2.waitKey(1)
                                    for i in range(1,1000):

                                        yield (b'--frame\r\n'
                                            
                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                                    # cv2.waitKey(1)
                                    # time.sleep(5)
                                    print("hi")
                                    paintWindow = np.zeros((480,640,3),np.uint8)+255
                                    paint = np.zeros((480,640,3),np.uint8)
                                    # print("hi")
                                    # result = {'status': 'failure','image':pred_digit}
                                    # return Response(letter_recognition(pred_digit),mimetype='multipart/x-mixed-replace; boundary=frame')
                                # res=json.dumps(result)


                                # return jsonify(result)
    
                               


                        cv2.rectangle(img,(x1,y1-5),(x2,y2+5),drawcolor,cv2.FILLED)    
                            
                        
                            
                    elif(finger[1] and finger[2]==False):
                        # fore_finger = (lmlist[8][0],lmlist[8][1])
                        # center = fore_finger
                        if   155<=y1<=395 and 210<=x1<=445:
                            if xp==0 and yp==0:
                                xp,yp=x1,y1
                            cv2.circle(img, (x1,y1), 10, drawcolor,cv2.FILLED)
                            if drawcolor==(0,0,0):
                                brush=50
                                cv2.line(img,(xp,yp),(x1,y1),drawcolor,brush)
                                
                                cv2.line(paint,(xp,yp),(x1,y1),drawcolor,brush)
                                cv2.line(paintWindow,(xp,yp),(x1,y1),(255,255,255),brush)
                            else:
                                cv2.line(img,(xp,yp),(x1,y1),drawcolor,15)
                                cv2.line(paintWindow,(xp,yp),(x1,y1),drawcolor,15)
                                cv2.line(paint,(xp,yp),(x1,y1),drawcolor,15)
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

        # if cv2.waitKey(1) == ord('q'):
        #     break
            # digitImagePath ='D:/service_now/EduKids/static/test_digits/'+str(pred_digit)+'.jpg'
            # print(img.shape)
            digitImagePath=os.getcwd()+'//static//test_digits//'+str(pred_digit)+'.png'
            digitImage=cv2.imread(digitImagePath)
            mergedImage = mergeImage(img,digitImage)
            ret, jpeg = cv2.imencode('.jpg', mergedImage)


           
            # frame=img

            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            
            # cv2.imshow("img",img)
            # # cv2.imshow("san",paintWindow)
            
            # cv2.waitKey(1)
        else:
            break
    # cap.release()
    # cv2.destroyAllWindows()
    cap.release()
    cv2.destroyAllWindows()
    # return -1;   
    # return render_template("login.html")    

def mergeImage(img1,img2):

    # img1 = cv2.resize(img1, (640, 440))
    # print(img1.shape)
    img2 = cv2.resize(img2, (480, 480))

    final=np.concatenate((img1,img2),axis=1)

    return final


def encrpt(password):
    # salt=bcrypt.genSaltSync()
    # hashed=bcrypt.hash(password,salt)
    hashed_password = hashlib.sha256(password.encode('utf-8')).hexdigest()
    return hashed_password

def verify(to_address):
    verify_code=str(randint(10000,99999))
    msg=MIMEMultipart()
    msg['to']=email.utils.formataddr(('Recipient',to_address))
    msg['from']=email.utils.formataddr(('EduKids','tarunajain109@gmail.com'))
    msg['Subject']='Email Verification Code from EduKids'
    body=f'Your verification code is {verify_code}.'
    msg.attach(MIMEText(body,'plain'))
    with smtplib.SMTP('smtp.gmail.com',587) as server:
        server.starttls()
        server.login('tarunajain109@gmail.com','ahbvmwmuhfyammtp')
        server.sendmail('tarunajain109@gmail.com',[to_address],msg.as_string())
    return verify_code  
def reset_pass(to_address):
    verify_code=str(randint(10000,99999))
    msg=MIMEMultipart()
    msg['to']=email.utils.formataddr(('Recipient',to_address))
    msg['from']=email.utils.formataddr(('EduKids','tarunajain109@gmail.com'))
    msg['Subject']='Reset Password Code from EduKids'
    body=f'Your verification code is {verify_code}.'
    msg.attach(MIMEText(body,'plain'))
    with smtplib.SMTP('smtp.gmail.com',587) as server:
        server.starttls()
        server.login('tarunajain109@gmail.com','ahbvmwmuhfyammtp')
        server.sendmail('tarunajain109@gmail.com',[to_address],msg.as_string())
    return verify_code  
@app.route('/digit')
def digit():
    return render_template("digit_recog.html")
@app.route('/digit_page')
def digit_page():
    # return render_template("letter_recog.html",msg='0')
    # while True:
    # redirect 
    return Response(letter_recognition(1),mimetype='multipart/x-mixed-replace; boundary=frame')
    # if x:
    #     return x
    # else:
    #     return render_template("index.html")
    # if(xx==-1):
    #     return render_template("letter_recog.html",msg='1')
    # elif xx==0:
    #     return render_template("letter_recog.html",msg='1')
    # else:
    #     return xx
            # return render_template("letter_recog.html",)  
        
@app.route('/')
def hello_world():
    # cap=cv2.VideoCapture(0)
    # return render_template("letter_recog.html",msg='0')
    return render_template("index.html") # home page

@app.route('/home')
def home(): #dashboard
    # if 'username' in session:
        # z=letter_recongnition(1)
        # if z==0:
        #     print("hi")
        # else:
        #     print("oops")
        # hand_tracker()

    return render_template('login.html',user=session['username'])

@app.route('/logout')
def logout():
    session.pop('username',None)
    return render_template('index.html')

@app.route('/login',methods=['GET','POST'])
def login():
    
    if request.method=='POST':
        username=request.form['user']
        password=request.form['passwd']

        cur.execute("select * from EDUKIDS where USERNAME='{}'".format(username))
        record=cur.fetchone()

        # print(record)
        if record:
            # hashed_password=record[1]
            if record[2]==0:
                msg='Please verify your mail'
            elif(hashlib.sha256(password.encode('utf-8')).hexdigest()==record[1]):
            # elif(bcrypt.checkpw(password,record[1])):
                # session.permanent=True
                session['username']=record[0]
                # print("hi")
                return redirect(url_for('home')) #dashboard
            else:
                msg= "Incorrect username or password"    
    # else:
    #     if "username" in session:
    #         return redirect(url_for('user'))
        
            
    return render_template('index.html')

@app.route("/user")
def user():
    # username=session['username']
    if "username" in session:
        # print("hiiiiiiiii")
        return redirect(url_for('home'))
    else:
        return render_template('index.html')  
  
@app.route('/validate',methods=['GET','POST'])
def validate():
    if request.method=='POST':
        if 'email' in session:
            # print("hi")j
            username=session['email']
            # print(username)
            cur.execute("update EDUKIDS set VERIFY={} where USERNAME='{}'".format(1,username))
            conn.commit()
            session.pop('email',None)
            # return redirect(url_for('user'))
            return render_template('index.html')
        else:
            flash("OTP NOT CORRECT")
            return render_template('verify.html')
    # if request.method=='POST':
@app.route('/final_reset',methods=['GET','POST'])
def final_reset():
    if request.method=='POST':
        if 'reset' in session:
            # print("hi")j
            passwd=encrpt(request.form['pass'])

            mail=session['reset']
            # print(username)
            cur.execute("update EDUKIDS set PASSWORD='{}' where EMAIL='{}'".format(passwd,mail))
            conn.commit()
            session.pop('reset',None)
            return redirect(url_for('hello_world'))
        
    # if request.method=='POST':
@app.route('/validate_new',methods=['GET','POST'])
def validate_new():
    if request.method=='POST':
        return render_template('changepass.html')
    # if request.method=='POST':
     
@app.route('/signin',methods=['GET','POST'])
def signin():
    msg=''
    if request.method=='POST':
        First_name=request.form['first']
        Second_name=request.form['second']
        username=request.form['user']
        password=encrpt(request.form['passwd'])
        age=int(request.form['age'])
        gender=request.form['gender']
        mail=request.form['mail']
        # salt=bcrypt.gensalt()
        # password=bcrypt.hashpw(passwordd.encode('utf-8'),salt)
        x=0
        # print("hi")
        cur.execute("select * from EDUKIDS where username='{}'".format(username))
        if cur.fetchone():
            flash("Username already used")
            return render_template('signin.html')
        # cur.execute(f"INSERT INTO EDUKIDS VALUES('{username}','{password}',{x},'{First_name}','{Second_name}',{age},'{gender}','{mail}')")
        sql="INSERT INTO EDUKIDS(USERNAME ,PASSWORD ,VERIFY ,FIRST ,SECOND ,AGE ,GENDER , EMAIL ) VALUES(:username,:password,:x,:First_name,:Second_name,:age,:gender,:mail)"
        cur.execute(sql,[username,password,x,First_name,Second_name,age,gender,mail])
        conn.commit()
        code=verify(mail) 
        # cur.execute("select * from EDUKIDS where username='{}' and password='{}'".format(username,password))
        session['email']=username
        return render_template('verify.html',code=code)
    else:
        flash("Enter Correctly")
        return render_template('signin.html') 

       
 
    #     record=cur.fetchone()
    #     # print(record)
    #     if record:
    #         session['loggedin']=True
    #         session['username']=record[0]
    #         return redirect(url_for('home'))
    #     else:
    #         msgg=message_flashed("Incorrect username or password")
    # return render_template('index.html',msg=msgg)
@app.route('/reset',methods=['GET','POST'])
def reset():
    msg=''
    if request.method=='POST':
        mail=request.form['mail']
        cur.execute("SELECT * FROM EDUKIDS WHERE EMAIL='{}'".format(mail))
        record=cur.fetchone()
        if(record):
            code=reset_pass(mail)
            session['reset']=mail
            return render_template('reset.html',code=code)


        else:
            msg='EMAIL NOT REGISTERED'
            return render_template('forgotpass.html',msg=msg)        
 
       

@app.route('/forget_pass')
def forget_pass():
    return render_template('forgotpass.html')
# @app.route('/hand_count')
# def hand_count():

if __name__=="__main__":
    app.run(debug=True)
