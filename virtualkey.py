import cv2
import mediapipe as mp
# cap=cv2.VideoCapture(0)
# cap.set(3,1200)
# cap.set(4,720)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # this is the magic!
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 900)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 700)
mphands = mp.solutions.hands
hands=mphands.Hands() # put here parameter that we needed detection con 0.8
mpDraw=mp.solutions.drawing_utils
keys=[["Q","W","E","R","T","Y","U","I","O","P"],["A","S","D","F","G","H","J","K","L"],
["Z","X","C","V","B","N","M"]]
def drawall(img,buttonlist):
    for button in buttonlist:
        x,y=button.pos
        w,h=button.size   
        cv2.rectangle(img,button.pos,(x+w,y+h),(255,0,255),cv2.FILLED)
        cv2.putText(img,button.text,(x+20,y+65),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),4)
    return img
   
class Button():
    def __init__ (self,pos,text,size=[85,85]):
        self.pos =pos
        self.size = size
        self.text=text
    # def draw(self,img): 
           # return img
buttonlist=[] 
for i in range(len(keys)):        
        for x,key in enumerate(keys[i]):    
            buttonlist.append(Button([100*x+50,100*i+50],key))         
         

while True:
    success,img=cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    lmlist=[]
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                h,w,c=img.shape
                cx,cy=int(lm.x*w) ,int(lm.y*h) 
                lmlist.append([id,cx,cy])
            mpDraw.draw_landmarks(img,handLms,mphands.HAND_CONNECTIONS)
    img=drawall(img,buttonlist) 
    if lmlist:
        for button in buttonlist:
            x,y=button.pos
            w,h=button.size
            print("hii",x,w)
            print(lmlist[8][0])
            if x<lmlist[8][0]<x+w:
                print("hi")
                cv2.rectangle(img,button.pos,(x+w,y+h),(0,255,0),cv2.FILLED)
                cv2.putText(img,button.text,(x+20,y+65),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),4)
  
    cv2.imshow("image",img)
    cv2.waitKey(1)
   