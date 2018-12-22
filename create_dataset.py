# FILENAME: create_dataset.py
# AUTHORS: Jalocha, Noga, Piekarski, Tancula

import cv2
from os import listdir
from os.path import isfile, join
import random
import pickle
import numpy as np


class CreateDataset:
    """
    Note:
        base dir should contain 'test' and 'train' subfolders
        image naming convention: initials followed by "_", rest part is insignificant,
                                example  DC_001.jpg (when it is Daniel Craig )
    """

    def __init__(self, base_dir, do_crop):
        self.base_dir = base_dir
        self.train_dataset = []
        self.test_data = []
        self.X = []
        self.y = []
        self.do_crop = do_crop
        self.initials2name = {
            "SC": 0,
            "PB": 1,
            "DC": 2,
            "RM": 3,
            "GL": 4,
            "TD": 5
        }
        self.train_face_size = 100  # declaring the size of a single training face for uniformity

    def read_test_data(self):
        path = self.base_dir + "/test"
        try:
            files = [f for f in listdir(path) if isfile(join(path, f))]
            for bond in files:
                tmp = cv2.imread(path + "/" + bond, cv2.IMREAD_GRAYSCALE)
                initials = bond.split("_")[0]
                self.test_data.append((tmp, self.initials2name[str(initials)]))
            random.shuffle(self.test_data)
        except Exception as e:
            print("Following error occured while reading test data: ", e)

    def read_training_data(self):
        path = self.base_dir + "/train"
        try:
            files = [f for f in listdir(path) if isfile(join(path, f))]
            for bond in files:
                tmp = cv2.imread(path + "/" + bond, cv2.IMREAD_GRAYSCALE)
                if self.do_crop:
                    self.crop_face(tmp, path + "/" + bond)
                if tmp.shape[0] != self.train_face_size or tmp.shape[1] != self.train_face_size:
                    tmp = cv2.resize(tmp, (self.train_face_size, self.train_face_size))
                initials = bond.split("_")[0]
                self.train_dataset.append((tmp, self.initials2name[str(initials)]))
            random.shuffle(self.train_dataset)
        except Exception as e:
            print("Following error occured while reading training data: ", e)

    def read_data(self):
        self.read_test_data()
        self.read_training_data()

    def crop_face(self, img_gray, path):
        haar = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        face = haar.detectMultiScale(img_gray, 1.2)
        for (x, y, w, h) in face:
            # cv2.imwrite(path, img[y:(y + w), x:(x + h)])
            img_gray = img_gray[y:(y + w), x:(x + h)]

    def create_model(self):
        for features, label in self.train_dataset:
            self.X.append(features)
            self.y.append(label)
        self.X = np.array(self.X).reshape(-1, self.train_face_size, self.train_face_size, 1)

    def save_training_to_pickle(self):  # save training model to pickle
        pickle_out = open("train_X.pickle", "wb")
        pickle.dump(self.X, pickle_out)
        pickle_out.close()
        pickle_out = open("train_y.pickle", "wb")
        pickle.dump(self.y, pickle_out)
        pickle_out.close()

    def prepare_training_dataset(self, do_save, do_augment):
        self.read_training_data()
        if do_augment:
            self.augment_train_data()
        self.create_model()
        if do_save:
            self.save_training_to_pickle()

    @staticmethod
    def flip_image(im):
        return np.fliplr(im)

    @staticmethod
    def translate_image(im, tx, ty):
        (height, width) = im.shape
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(im, M, (height, width))

    def augment_train_data(self):  # TODO: AUGMENT IN MORE SOPHISTICATED WAY EG. MORE FOR FEWER BONDS IN GIVEN CLASS
        train_data_tmp = self.train_dataset.copy()
        print("Augmenting data set - initial size {0}".format(len(train_data_tmp)))

        for im, label in self.train_dataset:
            train_data_tmp.append((self.flip_image(im), label))
            train_data_tmp.append((self.translate_image(im, 5, 0), label))
            train_data_tmp.append((self.translate_image(im, -5, 0), label))
            train_data_tmp.append((self.translate_image(im, 0, 5), label))
            train_data_tmp.append((self.translate_image(im, 0, -5), label))
            train_data_tmp.append((self.translate_image(im, 5, 5), label))
            train_data_tmp.append((self.translate_image(im, -5, -5), label))

        random.shuffle(train_data_tmp)
        self.train_dataset = train_data_tmp
        print("End of augmenting - final size {0}".format(len(train_data_tmp)))

