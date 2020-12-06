import time, math, sys
import tkinter as tk
from collections import deque
import cv2 as cv
from pyautogui import moveRel
import numpy as np
from PIL import ImageTk, Image


cv.startWindowThread()
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')


class CircularBuffer:

    def __init__(self, size):
        self.data = deque(maxlen=size)

    def append(self, x):
        self.data.append(x)

    def average(self):
        return sum(self.data) / len(self.data)

class Utils:
    def __init__(self):
        pass

    @staticmethod
    def toGray(img):
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    @staticmethod
    def showImg(img):
        cv.imshow('img', img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    @staticmethod
    def capturePicture(path=None):
        cap = cv.VideoCapture(0)
        ret, frame = cap.read()
        path = 'cap.jpeg' if not path else path
        cv.imwrite(path, frame)
        cap.release()
        return frame

    @staticmethod
    def captureVideo():
        cap = cv.VideoCapture(0)
        print('Press q to close window!')
        while True:
            ret, frame = cap.read()
            rects = eye_cascade.detectMultiScale(frame, 1.3, 5)
            cv.imshow('frame',frame)
            i = 0
            for x,y,w,h in rects:
                cv.imshow('roi'+str(i), frame[y:y+h, x:x+w])
                i += 1
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
            

def drawRectangles(img, rects):
    for x, y, w, h in rects:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return img

def roiRect(img, rects):
    ''' Returns first roi
    '''
    rois = []
    for x, y, w, h in rects:
        rois.append(img[y:y+h, x:x+w])
    return rois


DETECT = {
    'face' : False,
    'eyes' : False,
    'pupils': False
}

def toggleFace():
    DETECT['face'] = not DETECT['face']
def toggleEyes():
    DETECT['eyes'] = not DETECT['eyes']
def togglePupils():
    DETECT['pupils'] = not DETECT['pupils']    

window = tk.Tk()
frame = tk.Frame(window)
frame.pack()
label = tk.Label(frame)
label.pack()

faceButton = tk.Button(frame,text='Toggle Face Detection', command=toggleFace)
faceButton.pack()
eyesButton = tk.Button(frame,text='Toggle Eyes Detection', command=toggleEyes)
eyesButton.pack()
pupilButton = tk.Button(frame,text='Toggle Pupils Detection', command=togglePupils)
pupilButton.pack()

cap = cv.VideoCapture(0)
av_x = CircularBuffer(5)
av_y = CircularBuffer(5)


def pipeline(image):
    frame = Utils.toGray(image)    
    rec = face_cascade.detectMultiScale(frame, 1.3, 3)
    rects = eye_cascade.detectMultiScale(frame, 1.3, 5)
    for x, y, w, h in rects:
        roi = frame[y:y+h, x:x+w]
        gray = cv.GaussianBlur(roi,(3,3), 0)
        threshold = gray
        ret, threshold = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        circles = cv.HoughCircles(threshold,cv.HOUGH_GRADIENT,1,200,param1=200,param2=1,minRadius=7,maxRadius=13)    
        #print(circles)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles:
                # Draw circles on roi
                if DETECT['pupils']:
                    cv.circle(image,(i[0,0]+x,i[0,1]+y),int(i[0,2]),(0,255,0),1)
                    cv.circle(image,(i[0,0]+x,i[0,1]+y),2,(0,0,255),3)
                av_x.append(i[0,0]+x)
                av_y.append(i[0,1]+y)

    if DETECT['eyes']:
        drawRectangles(image, rects)
    if DETECT['face']:
        drawRectangles(image, rec)
    return image

def stream():
    _, frame = cap.read()
    frame = pipeline(frame)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(1, stream)


stream()
window.mainloop()



# epsilon = 10

# stableX, stableY = 0, 0

# isCalibrating = True

# while True:
#     ret, original = cap.read()
#     frame = Utils.toGray(original)
#     rec = face_cascade.detectMultiScale(frame, 1.3, 3)
#     rects = eye_cascade.detectMultiScale(frame, 1.3, 5)
#     for x, y, w, h in rects:
#         roi = frame[y:y+h, x:x+w]
#         gray = cv.GaussianBlur(roi,(3,3), 0)
#         threshold = gray
#         ret, threshold = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
#         circles = cv.HoughCircles(threshold,cv.HOUGH_GRADIENT,1,200,param1=200,param2=1,minRadius=7,maxRadius=13)    
#         #print(circles)
#         if circles is not None:
#             circles = np.uint16(np.around(circles))
#             for i in circles:
#                 # Draw circles on roi
#                 cv.circle(original,(i[0,0]+x,i[0,1]+y),int(i[0,2]),(0,255,0),1)
#                 cv.circle(original,(i[0,0]+x,i[0,1]+y),2,(0,0,255),3)
#                 av_x.append(i[0,0]+x)
#                 av_y.append(i[0,1]+y)

#     drawRectangles(original, rects)
#     drawRectangles(original, rec)

#     if not isCalibrating:
#         new_average_x = av_x.average()
#         new_average_y = av_y.average()

#         if abs(new_average_x - stableX) > epsilon:
#             s = new_average_x - stableX
#             factor = -1.5
#             if s > 0:
#                 moveRel(factor*s, 0)
#             if s < 0:
#                 moveRel(factor*s, 0)
#             stableX = new_average_x

#     cv.imshow('live', original)
#     key = cv.waitKey(2)
#     if key == ord('q'):
#         break
#     elif key == ord('d'):
#         isCalibrating = False
#         stableX = av_x.average()
#         stableY = av_y.average()
#         print('X:',stableX)
#         print('Y:',stableY)
#     elif key == ord('a'):
#         print(f'Average: {av_x.average()}, {av_y.average()}')

# cap.release()
# cv.destroyAllWindows()
