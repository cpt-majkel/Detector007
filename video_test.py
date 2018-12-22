import pickle
import cv2
import numpy as np
import tensorflow as tf
from create_dataset import CreateDataset

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
    names = ["SC","PB","DC","RM","GL","TD"]
    #base_image_dir = 'input_data'
    #dataset = CreateDataset(base_image_dir, False)
    pickle_classifier = open("classifier_NN.pickle", "rb")
    classifier = pickle.load(pickle_classifier)
    img = cv2.imread('all_bonds.jpg')#('input_data/test/RM_001.jpg')
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    face = haar.detectMultiScale(img_gray, 1.1)
    for (x, y, w, h) in face:
        tmp = img_gray[y:(y + w), x:(x + h)]
        tmp1 = cv2.resize(tmp, (100, 100))
        tmp1 = tmp1 / 255.0
        input = tmp1[np.newaxis,...,np.newaxis] #adding axes to predict properly
        prediction = classifier.predict(input)
        which = np.argmax(prediction)

        print(which, prediction)
        cv2.rectangle(img, (x-30,y-30),(x+w+30,y+h+30), (255, 255, 0), thickness=3)
        cv2.rectangle(img,(x-30,y-70),(x+w,y-30),(255,255,0),cv2.FILLED)
        cv2.putText(img, names[which], (x-25, y-40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,255), lineType=cv2.LINE_AA)
    cv2.imshow('test',img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

#test_video()
image_testing()
