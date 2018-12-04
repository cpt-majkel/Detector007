# FILENAME: create_dataset.py
# AUTHORS: Jalocha, Noga, Piekarski, Tancula

import cv2
from os import listdir
from os.path import isfile, join
import random
import pickle

class CreateDataset:
    """
    Note:
        base dir should contain 'test' and 'train' subfolders
        image naming convention: initials followed by "_", rest part is insignificant,
                                example  DC_001.jpg (when it is Daniel Craig )
    """

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.train_dataset = []
        self.test_data = []
        self.initials2name = {
            "SC": "Sean Connery",
            "PB": "Pierce Brosnan",
            "DC": "Daniel Craig",
            "RM": "Roger Moore",
            "GL": "George Lazenby",
            "TD": "Timothy Dalton"
        }
        self.train_face_size = 100 #declaring the size of a single training face for uniformity

    def read_test_data(self):
        path = self.base_dir + "/test"
        try:
            files = [f for f in listdir(path) if isfile(join(path, f))]
            for bond in files:
                tmp = cv2.imread(path + "/" + bond, cv2.IMREAD_GRAYSCALE)
                initials = bond.split("_")[0]
                self.test_data.append((tmp, self.initials2name[str(initials)]))
        except Exception as e:
            print("Following error occured while reading test data: ", e)

    def read_training_data(self):
        path = self.base_dir + "/train"
        try:
            files = [f for f in listdir(path) if isfile(join(path, f))]
            for bond in files:
                tmp = cv2.imread(path + "/" + bond, cv2.IMREAD_GRAYSCALE)
                tmp_path = path + "/" + bond
                #ret = self.crop_face(tmp, tmp_path)
                #ret_grayscale = cv2.cvtColor(ret,cv2.COLOR_BGR2GRAY)
                initials = bond.split("_")[0]
                self.train_dataset.append((tmp, self.initials2name[str(initials)]))
            random.shuffle(self.train_dataset)
        except Exception as e:
            print("Following error occured while reading training data: ", e)

    def read_data(self):
        self.read_test_data()
        self.read_training_data()


    def crop_face(self, path):
        haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        try:
            files = [f for f in listdir(path) if isfile(join(path, f))]
            for bond in files:
                img1 = []
                print path + "/" + bond
                tmp = cv2.imread(path + "/" + bond)
                tmp_path = path + "/" + bond
                img_gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
                face = haar.detectMultiScale(img_gray, 1.2)
                for (x, y, w, h) in face:
                    # cv2.imwrite(path, img[y:(y + w), x:(x + h)])
                    img1 = tmp[y:(y + w), x:(x + h)]
                    self.resize_faces(img1, tmp_path)
        except Exception as e:
            print("Following error occured while reading training data: ", e)

    def resize_faces(self, img, path):
        if img.shape[0] < self.train_face_size or img.shape[1] < self.train_face_size:
            pass
        else:
            img = cv2.resize(img,(self.train_face_size,self.train_face_size))
            cv2.imwrite(path,cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))

    def save_training_to_pickle(self, data, labels): #save training model to pickle
        pickle_out = open("train_X.pickle", "wb")
        pickle.dump(data, pickle_out)
        pickle_out.close()
        pickle_out = open("train_y.pickle", "wb")
        pickle.dump(labels, pickle_out)
        pickle_out.close()

    def load_training_data_pickle(self): #load pickle training model
        pickle_data = open("train_X.pickle", "rb")
        X = pickle.load(pickle_data)
        pickle_labels = open("train_y.pickle", "rb")
        y = pickle.load(pickle_labels)
        return X,y  #returns data and labels, in that order