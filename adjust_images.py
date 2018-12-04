# FILENAME: adjust_images.py
# AUTHORS: Jalocha, Noga, Piekarski, Tancula

from create_dataset import CreateDataset
import cv2
import numpy as np
import time

"""Script made to be run once, only to prepare the training set and save it"""

def main():

    base_image_dir = 'input_data'
    dataset = CreateDataset(base_image_dir)
    #dataset.crop_face(dataset.base_dir + "/train")
    dataset.read_data()
    labels_training = []
    data_training = []
    for elem in dataset.train_dataset: labels_training.append(elem[1])
    for dat in dataset.train_dataset: data_training.append(dat[0])

    #print data_training[0].shape

    # for dat in np.array(data_training):
    #     print dat.shape
    #     dat = dat.reshape(dataset.train_face_size, dataset.train_face_size, 1)

    #print data_training[0].reshape(-1, 100, 100, 1)

    #ZJEBALO SIE
    #data_training = np.array(data_training).reshape(-1, dataset.train_face_size, dataset.train_face_size, 1)

    print data_training[0]

    for x in range(0,189):
        data_training[x] = np.array(data_training[x]).reshape(100,100,1)
    print data_training[189]




    #print data_training.reshape(100,100,1)

    #dataset.save_training_to_pickle(data_training,labels_training)

    #print data_training.shape([1])

if __name__ == "__main__":
    main()
