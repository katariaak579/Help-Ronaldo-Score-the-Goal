import cv2
from cv2 import FONT_HERSHEY_DUPLEX
import numpy as np
from cv2 import COLOR_BGR2GRAY
from cv2 import THRESH_BINARY
from collections import deque


# Classes Used 
class BgExtract:
    def __init__(self,width,height,scale,maxlen=10): 
        self.maxlen=maxlen
        self.width=width//scale
        self.scale=scale
        self.height=height//scale
        self.buffer=deque(maxlen=maxlen)
        self.bg=None
    
    def cal_if_notfull(self):
        self.bg=np.zeros((self.height,self.width,),dtype='float32')
        for i in self.buffer:
            self.bg+=i
        self.bg//=len(self.buffer)

    def cal_if_full(self,old,new):
        self.bg-=old/self.maxlen
        self.bg+=new/self.maxlen

    def add_frame(self,frame):
        if self.maxlen>len(self.buffer):
            self.buffer.append(frame)
            self.cal_if_notfull()
        else:
            old=self.buffer.popleft()
            self.buffer.append(frame)
            self.cal_if_full(old,frame)

    def output_frame(self):       
        return self.bg.astype('uint8')

    def apply(self,frame):
        down_scale=cv2.resize(frame,(self.width,self.height))
        gray=cv2.cvtColor(down_scale,COLOR_BGR2GRAY)
        gray=cv2.GaussianBlur(gray,(5,5),0)
    
        self.add_frame(gray)
        absdifference=cv2.absdiff(gray,self.output_frame())
        _,maskabs=cv2.threshold(absdifference,15,255,THRESH_BINARY)
        return cv2.resize(maskabs,(self.width*self.scale,self.height*self.scale))

class Game:
    def __init__(self,width,height,size=50,object_image="Images/Runner.jpg"):
        self.width=width
        self.height=height
        self.size=size
        self.img=cv2.imread(object_image)
        self.img=cv2.resize(self.img,(self.size,self.size))
        gray=cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)        
        self.mask=cv2.threshold(gray,1,255,cv2.THRESH_BINARY)[1]
        self.x=np.random.randint(0,self.width-self.size)
        self.y=np.random.randint(0,self.height-self.size)

    def add_frame(self,frame):
        roi=frame[self.y:self.y+self.size,self.x:self.x+self.size]
        roi[np.where(self.mask)]=0
        roi+=self.img


class PlayerObject(Game):
    def update_pos(self,fg_frame):
        roi=fg_frame[self.y:self.y+self.size,self.x:self.x+self.size]

        check_if_it=np.any(roi[np.where(self.mask)])
        if check_if_it:           
            best_fit=np.inf
            best_x=0
            best_y=0

            for _ in range(8):
                x_bar=np.random.randint(-15,16)
                y_bar=np.random.randint(-15,16)

                if self.x+x_bar<0 or self.x + self.size+ x_bar>=self.width or self.y+y_bar<0 or self.y + self.size + y_bar>=self.height:
                    continue
                
                roi=fg_frame[self.y+y_bar:self.y+y_bar+self.size,self.x+x_bar:self.x+x_bar+self.size]
                overlap=np.count_nonzero(roi[np.where(self.mask)])
                if overlap<best_fit:
                    best_fit=overlap
                    best_x=x_bar
                    best_y=y_bar
            
            self.x+=best_x
            self.y+=best_y


        return check_if_it
    

class Gaming:
    def __init__(self, width, height):
        self.width=width
        self.height=height
        self.score = 0
        self.player = PlayerObject(width, height)
        self.goal=Game(width,height,object_image="Images/Goal.jpg")
        self.tracker=TrackerObject(width,height,self.player)
        self.hit=False
 
    def update_pos(self, fg_frame):
        self.player.update_pos(fg_frame)
        self.hit=self.tracker.update_pos()


 
    def add_frame(self, frame):
        self.goal.add_frame(frame)
        self.player.add_frame(frame)
        self.tracker.add_frame(frame)
        if abs(self.player.x + self.player.size//2 - (self.goal.x + self.goal.size//2)) < self.player.size//2 and abs(self.player.y + self.player.size//2 - (self.goal.y + self.goal.size//2)) < self.player.size//2:
            self.score += 1
            self.goal.x = np.random.randint(0, self.width - self.goal.size)
            self.goal.y = np.random.randint(0, self.height - self.goal.size)
            frame[:,:,1] = 255  
        
        if self.hit:
            self.score-=1
            self.player.x=np.random.randint(0,self.width-self.player.size)
            self.player.y=np.random.randint(0,self.height-self.player.size)
            frame[:,:,2]=255

class TrackerObject(Game):
    def __init__(self, width, height, follow=None):
        Game.__init__(self, width, height, size=25, object_image="Images/tracker.jpg")
        self.follow = follow
        self.speed_x = 0
        self.speed_y = 0
        self.max_speed = 1
 
    def update_pos(self):
        diff_x = self.follow.x + self.follow.size//2 - (self.x + self.size//2)
        diff_y = self.follow.y + self.follow.size//2 - (self.y + self.size//2)
 
        if abs(diff_x) < self.size and abs(diff_y) < self.size:
            return True
 
        if diff_x < 0 and not self.speed_x < -self.max_speed:
            self.speed_x -= np.random.randint(0, 2)
        elif diff_x > 0 and not self.speed_x > self.max_speed:
            self.speed_x += np.random.randint(0, 2)
        if diff_y < 0 and not self.speed_y < -self.max_speed:
            self.speed_y -= np.random.randint(0, 2)
        elif diff_y > 0 and not self.speed_y > self.max_speed:
            self.speed_y += np.random.randint(0, 2)
 
        if self.x + self.speed_x < 0 or self.x + self.speed_x + self.size >= self.width or \
                self.y + self.speed_y < 0 or self.y + self.speed_y + self.size >= self.height:
            return False
 
        self.x += self.speed_x
        self.y += self.speed_y
        return False


# Variables 
width=640
height=480
scale_down=2


cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)

bg_buffer=BgExtract(width,height,scale_down,maxlen=5)
game=Gaming(width,height)

while True:
    _,frame=cap.read()
    frame=cv2.flip(frame,1)
    fg_frame=bg_buffer.apply(frame)
    game.update_pos(fg_frame)
    game.add_frame(frame)

    text = f"Score: {game.score}"
    cv2.putText(frame,text,(10,30),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,0,0),2)
    cv2.imshow("Game",fg_frame)
    cv2.imshow("Actual",frame)
   
    if cv2.waitKey(1) == ord('q'):
        break
