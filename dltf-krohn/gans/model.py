import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, BatchNormalization, Dropout, Flatten
from keras.layers import Activation, Reshape, Conv2DTranspose, UpSampling2D  # new!
from keras.optimizers import RMSprop

from discriminator import discriminator_builder
from generator import generator_builder
from data_manager import get_data


def adversarial_builder(discriminator, generator, z_dim=100):
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.0004, decay=3e-8, clipvalue=1.0),
                  metrics=['accuracy'])
    # model.summary()
    return model


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


def train(epochs=2000, batch=128):
    d_metrics = []
    a_metrics = []

    running_d_loss = 0
    running_d_acc = 0
    running_a_loss = 0
    running_a_acc = 0

    for i in range(epochs):

        if i % 10 == 0:
            print(i)

        real_imgs = np.reshape(data[np.random.choice(data.shape[0], batch, replace=False)], (batch, 28, 28, 1))
        fake_imgs = generator.predict(np.random.uniform(-1.0, 1.0, size=[batch, 100]))

        x = np.concatenate((real_imgs, fake_imgs))
        y = np.ones([2 * batch, 1])
        y[batch:, :] = 0

        make_trainable(discriminator, True)

        d_metrics.append(discriminator.train_on_batch(x, y))
        running_d_loss += d_metrics[-1][0]
        running_d_acc += d_metrics[-1][1]

        make_trainable(discriminator, False)

        noise = np.random.uniform(-1.0, 1.0, size=[batch, 100])
        y = np.ones([batch, 1])

        a_metrics.append(adversarial_model.train_on_batch(noise, y))
        running_a_loss += a_metrics[-1][0]
        running_a_acc += a_metrics[-1][1]

        if (i + 1) % 500 == 0:

            print('Epoch #{}'.format(i + 1))
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, running_d_loss / i, running_d_acc / i)
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, running_a_loss / i, running_a_acc / i)
            print(log_mesg)

            noise = np.random.uniform(-1.0, 1.0, size=[16, 100])
            gen_imgs = generator.predict(noise)
            # np.save('gen_img' + str(i+1) + '.npy', gen_imgs)

            plt.figure(figsize=(5, 5))

            for k in range(gen_imgs.shape[0]):
                plt.subplot(4, 4, k + 1)
                plt.imshow(gen_imgs[k, :, :, 0], cmap='gray')
                plt.axis('off')

            plt.tight_layout()
            plt.show()
            # plt.savefig('apple' + str(i+1) + '.png')

    return a_metrics, d_metrics


if __name__ == '__main__':
    data = get_data()
    discriminator = discriminator_builder()
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=RMSprop(lr=0.0008, decay=6e-8, clipvalue=1.0),
                          metrics=['accuracy'])
    generator = generator_builder()
    adversarial_model = adversarial_builder(discriminator, generator)
    a_metrics_complete, d_metrics_complete = train(epochs=3000)
