import pandas as pd
import os

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


class DataManager:
    def __init__(self,
                 data_dir='~/data/mnist_kaggle',
                 random_state=42,
                 test_size=.1):
        self.data_dir = data_dir
        self.train_file = 'train.csv'
        self.test_file = 'test.csv'
        self.random_state = random_state
        self.test_size = test_size

    def get_train_data(self):
        filename = os.path.join(self.data_dir, self.train_file)
        df = pd.read_csv(filename)
        x_train = df.iloc[:, 1:].values
        y_train = df['label'].values

        # preprocess data
        x_train = x_train.reshape((42000, 28, 28, 1))
        x_train = x_train.astype('float32') / 255

        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                          test_size=self.test_size,
                                                          random_state=self.random_state)

        y_train = to_categorical(y_train)
        y_val = to_categorical(y_val)

        return x_train, x_val, y_train, y_val

    def get_test_data(self):
        filename = os.path.join(self.data_dir, self.test_file)
        df = pd.read_csv(filename)
        x_test = df.values

        # preprocess data
        x_test = x_test.reshape((28000, 28, 28, 1))
        x_test = x_test.astype('float32') / 255

        return x_test
