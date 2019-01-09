# FILENAME: main.py
# AUTHORS: Jalocha, Noga, Piekarski, Tancula

from create_dataset import CreateDataset
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


def main():
    base_image_dir = 'input_data'
    do_augment = True
    validation_split = 0.8 if do_augment else 0.2
    dataset = CreateDataset(base_image_dir, False)
    dataset.prepare_training_dataset(True, do_augment)
    X, y = load_training_data_pickle('train')
    X = X/255.0

    if do_augment:
        X_orig, y_orig = load_training_data_pickle('before_augment')
        val_ind = int(np.round((1-validation_split) * len(X_orig)))
        X_val = X_orig[val_ind:len(X_orig)-1]
        y_val = y_orig[val_ind:len(y_orig)-1]
        X_val = X_val/255.0
        delete_indices = np.where(np.in1d(X, X_val))
        np.delete(X, delete_indices)
        np.delete(y, delete_indices)
    else:
        val_ind = int(np.round((1 - validation_split) * len(X)))
        X_val = X[val_ind:len(X) - 1]
        y_val = y[val_ind:len(y)-1]
        X = X[0:val_ind]
        y = y[0:val_ind]

    print(len(X), val_ind)

    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=(100, 100, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    # model.add(Dense(64))
    model.add(Dense(6))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(X, y, batch_size=2, epochs=5, validation_data=(X_val, y_val))
    pickle_out = open("classifier_NN.pickle", "wb")
    pickle.dump(model, pickle_out)


def load_training_data_pickle(pickle_name):  # load pickle training model
    pickle_data = open(pickle_name + "_X.pickle", "rb")
    X = pickle.load(pickle_data)
    pickle_labels = open(pickle_name + "_y.pickle", "rb")
    y = pickle.load(pickle_labels)
    return X, y  # returns data and labels, in that order


if __name__ == "__main__":
    main()
