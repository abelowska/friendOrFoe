from keras.preprocessing.image import ImageDataGenerator


class DataGenerator:

    @staticmethod
    def get_train_set():
        train_data_gen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        training_set = train_data_gen.flow_from_directory(
            '../../data/train',
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary')

        return training_set

    @staticmethod
    def get_test_set():

        test_data_gen = ImageDataGenerator(rescale=1. / 255)

        test_set = test_data_gen.flow_from_directory(
            '../../data/test',
            target_size=(64, 64),
            batch_size=32,
            class_mode='binary')

        return test_set
