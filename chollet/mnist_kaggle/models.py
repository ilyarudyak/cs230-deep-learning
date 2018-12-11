import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator

from datetime import datetime

from utils import *


class MnistModel:
    def __init__(self,
                 name='krohn',
                 data_dir=None,
                 epochs=5,
                 batch_size=64,
                 validation_split=.2,
                 optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics='accuracy',
                 verbose=1):
        self.model = Sequential()
        if name == 'chollet':
            self.build_chollet_model()
            self.log_dir = 'logs/chollet'
        elif name == 'krohn':
            self.build_krohn_model()
            self.log_dir = 'logs/krohn'

        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split

        if data_dir:
            dm = DataManager(data_dir=data_dir)
        else:
            dm = DataManager()

        self.x_train, self.x_val, self.y_train, self.y_val = dm.get_train_data()
        self.x_test = dm.get_test_data()

        self.optimizer = optimizer
        self.loss = loss
        self.metrics = [metrics]
        self.tensorbd = [TensorBoard(self.log_dir)]

        date = datetime.today().strftime('%Y%m%d_%H%M')
        self.submission_filename = f'submissions/submission_{date}.csv'

        self.verbose = verbose

    def build_chollet_model(self):
        self.model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(units=64, activation='relu'))
        self.model.add(Dense(units=10, activation='softmax'))

    def build_krohn_model(self):
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(10, activation='softmax'))

    def train_model(self):
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)

        history = self.model.fit(x=self.x_train, y=self.y_train,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 validation_data=(self.x_val, self.y_val),
                                 callbacks=self.tensorbd,
                                 verbose=self.verbose)
        return history

    def train_model_aug(self):
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=self.metrics)

        dg = ImageDataGenerator(rotation_range=10,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                shear_range=0.1,
                                zoom_range=0.1)
        dg.fit(self.x_train)

        history = self.model.fit_generator(dg.flow(self.x_train, self.y_train,
                                                   batch_size=self.batch_size),
                                           epochs=self.epochs,
                                           validation_data=(self.x_val, self.y_val),
                                           steps_per_epoch=self.x_train.shape[0] // self.batch_size,
                                           callbacks=self.tensorbd,
                                           verbose=self.verbose)
        return history

    def make_submission(self):
        y_eval = self.model.predict(x=self.x_test,
                                    batch_size=self.batch_size)
        # y_eval one-hot-encoded (28000, 10)
        # we have to convert into (28000,)
        y_classes = np.argmax(y_eval, axis=1)
        d = {'ImageId': np.arange(1, y_classes.shape[0] + 1),
             'Label': y_classes}
        df = pd.DataFrame(d)
        df.to_csv(self.submission_filename, index=False)


if __name__ == '__main__':
    mm = MnistModel(name='chollet', epochs=1, batch_size=512, verbose=0)
    history = mm.train_model()
    print(history)
    # mm.make_submission()
