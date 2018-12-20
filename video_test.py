import pickle
import cv2
import numpy as np

def test_video():
    cap = cv2.VideoCapture('bond2.mp4')

    if(cap.isOpened() == False):
        print("Error loading file\n")

    while cap.isOpened():
        ret, frame = cap.read()

        if ret == True:
            draw_rectangle_detected(frame)
            cv2.imshow('Bond Classifier Test', frame)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()

def draw_rectangle_detected(frame):
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    face = haar.detectMultiScale(frame_gray, 1.1)
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x-30,y-30), (x+w+30,y+h+30), (0,255,0), thickness=3)
        cv2.rectangle(frame,(x-30,y-70),(x+w,y-30),(255,255,0),cv2.FILLED)
        cv2.putText(frame, 'Sample', (x-25, y-40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,255), lineType=cv2.LINE_AA)
def image_testing():
    img = cv2.imread('facetest.jpg')
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    face = haar.detectMultiScale(img_gray, 1.1)
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x-30,y-30),(x+w+30,y+h+30), (255, 255, 0), thickness=3)
        cv2.rectangle(img,(x-30,y-70),(x+w,y-30),(255,255,0),cv2.FILLED)
        cv2.putText(img, 'Sample', (x-25, y-40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,255), lineType=cv2.LINE_AA)
    cv2.imshow('test',img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

# test_video()
image_testing()
