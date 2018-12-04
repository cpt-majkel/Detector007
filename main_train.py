# FILENAME: main.py
# AUTHORS: Jalocha, Noga, Piekarski, Tancula

from create_dataset import CreateDataset
import cv2
import tensorflow as tf

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import pickle

def main():

    base_image_dir = 'input_data'
    dataset = CreateDataset(base_image_dir)
    #cv2.imshow(dataset.train_dataset[5][1], dataset.train_dataset[5][0])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    train_data, train_labels = dataset.load_training_data_pickle()

    print train_data[0].shape[0:]

    model = Sequential()

    model.add(Conv2D(256,(3,3), input_shape=train_data[0].shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(64))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(train_data, train_labels, batch_size=32, epochs=3, validation_split=0.3)

if __name__ == "__main__":
    main()
