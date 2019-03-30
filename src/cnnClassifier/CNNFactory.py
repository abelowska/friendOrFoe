from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


class CNNFactory:

    @staticmethod
    def create():
        classifier = Sequential()

        classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation="relu"))

        classifier.add(MaxPooling2D(pool_size=(2, 2)))

        classifier.add(Flatten())

        classifier.add(Dense(activation="relu", units=128))
        classifier.add(Dense(activation="sigmoid", units=1))

        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        return classifier
