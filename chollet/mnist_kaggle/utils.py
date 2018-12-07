import pandas as pd
import os

from keras.utils import to_categorical


class DataManager:
    def __init__(self, data_dir='/Users/ilyarudyak/data/mnist_kaggle'):
        self.data_dir = data_dir
        self.train_file = 'train.csv'
        self.test_file = 'test.csv'

    def get_train_data(self):
        filename = os.path.join(self.data_dir, self.train_file)
        df = pd.read_csv(filename)
        x_train = df.iloc[:, 1:].values
        y_train = df['label'].values

        # preprocess data
        x_train = x_train.reshape((42000, 28, 28, 1))
        x_train = x_train.astype('float32') / 255
        y_train = to_categorical(y_train)

        return x_train, y_train

    def get_test_data(self):
        filename = os.path.join(self.data_dir, self.test_file)
        df = pd.read_csv(filename)
        x_test = df.values

        # preprocess data
        x_test = x_test.reshape((28000, 28, 28, 1))
        x_test = x_test.astype('float32') / 255

        return x_test
