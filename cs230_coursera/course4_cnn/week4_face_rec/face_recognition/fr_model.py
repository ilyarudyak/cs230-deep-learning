from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
import cv2
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import time
from fr_utils import *
from inception_blocks_v2 import *
from triplet_loss import triplet_loss


def load_model():
    start = time.time()
    print('start loading model ...')
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])

    start2 = time.time()
    print(f'elapsed: {start2 - start} start loading weights ... ')
    load_weights_from_FaceNet(FRmodel)

    finish = time.time()
    print(f'elapsed: {finish - start2} finish loading model ...')
    return FRmodel


def build_database(FRmodel):
    print('start building db ...')
    database = {"danielle": img_to_encoding("images/danielle.png", FRmodel),
                "younes": img_to_encoding("images/younes.jpg", FRmodel),
                "tian": img_to_encoding("images/tian.jpg", FRmodel),
                "andrew": img_to_encoding("images/andrew.jpg", FRmodel),
                "kian": img_to_encoding("images/kian.jpg", FRmodel),
                "dan": img_to_encoding("images/dan.jpg", FRmodel),
                "sebastiano": img_to_encoding("images/sebastiano.jpg", FRmodel),
                "bertrand": img_to_encoding("images/bertrand.jpg", FRmodel),
                "kevin": img_to_encoding("images/kevin.jpg", FRmodel),
                "felix": img_to_encoding("images/felix.jpg", FRmodel),
                "benoit": img_to_encoding("images/benoit.jpg", FRmodel),
                "arnaud": img_to_encoding("images/arnaud.jpg", FRmodel)}
    print('finish building db ...')
    return database


def verify(image_path, identity, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity".

    Arguments:
    image_path -- path to an image
    identity -- string, name of the person you'd like to verify the identity. Has to be a resident of the Happy house.
    database -- python dictionary mapping names of allowed people's names (strings) to their encodings (vectors).
    model -- your Inception model instance in Keras

    Returns:
    dist -- distance between the image_path and the image of "identity" in the database.
    door_open -- True, if the door should open. False otherwise.
    """

    ### START CODE HERE ###

    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
    print('start verification ...')
    encoding = img_to_encoding(image_path, model)

    # Step 2: Compute distance with identity's image (≈ 1 line)
    dist = np.linalg.norm(encoding - database[identity])

    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if dist < 0.7:
        print("It's " + str(identity) + ", welcome home!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False

    ### END CODE HERE ###

    return dist, door_open


if __name__ == '__main__':
    K.set_image_data_format('channels_first')
    FRmodel = load_model()
    database = build_database(FRmodel)
    print(database['dan'].shape)
    dist, door_open = verify("images/camera_0.jpg", "younes", database, FRmodel)
    print(dist, door_open)
