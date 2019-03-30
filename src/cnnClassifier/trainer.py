from src.cnnClassifier.CNNFactory import CNNFactory


class CNNTrainer:
    def __init__(self, training_set, test_set):
        self.training_set = training_set
        self.test_set = test_set

    def train(self, classifier=CNNFactory.create()):
        # training_set = DataGenerator.get_train_set()
        # test_set = DataGenerator.get_test_set()

        classifier.fit_generator(
            self.training_set,
            steps_per_epoch=2000,
            epochs=3,
            validation_data=self.test_set,
            validation_steps=200)

        return classifier
