# FILENAME: main.py
# AUTHORS: Jalocha, Noga, Piekarski, Tancula

from create_dataset import CreateDataset
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelEncoder


def main():
    base_image_dir = 'input_data'
    dataset = CreateDataset(base_image_dir, False)
    dataset.prepare_training_dataset(True)
    X, y = load_training_data_pickle()
    X = X/255.0
    labelencoder_y_1 = LabelEncoder()
    y_train = labelencoder_y_1.fit_transform(y)
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Dense(6))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model.fit(X, y_train, batch_size=2, epochs=3, validation_split=0.2)


def load_training_data_pickle():  # load pickle training model
    pickle_data = open("train_X.pickle", "rb")
    X = pickle.load(pickle_data)
    pickle_labels = open("train_y.pickle", "rb")
    y = pickle.load(pickle_labels)
    return X, y  # returns data and labels, in that order


if __name__ == "__main__":
    main()
