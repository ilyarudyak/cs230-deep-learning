from keras.engine.saving import load_model, model_from_json
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

MODEL_WEIGHTS = 'saved_models/FRmodel_weights.h5'
MODEL_ARCH = 'saved_models/FRmodel_arch.h5'


# noinspection PyShadowingNames
def build_model():
    # this is an Inception model from file inception_blocks_v2.py
    # build with keras functional API
    FRmodel = faceRecoModel(input_shape=(3, 96, 96))
    FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
    load_weights_from_FaceNet(FRmodel)

    # save model to disk (Weights + Model Architecture)
    FRmodel.save_weights(MODEL_WEIGHTS)
    with open(MODEL_ARCH, 'w') as f:
        f.write(FRmodel.to_json())


def load_model_from_disk():
    with open(MODEL_ARCH, 'r') as f:
        FRmodel = model_from_json(f.read())
        FRmodel.load_weights(MODEL_WEIGHTS)
    return FRmodel


# noinspection PyShadowingNames
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


# noinspection PyShadowingNames
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


def who_is_it(image_path, database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.

    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras

    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """

    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding = img_to_encoding(image_path, model)

    ## Step 2: Find the closest encoding ##

    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(encoding - db_enc)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity


if __name__ == '__main__':
    K.set_image_data_format('channels_first')
    # build_model()

    FRmodel = load_model_from_disk()
    database = build_database(FRmodel)

    # dist, door_open = verify("images/camera_0.jpg", "younes", database, FRmodel)
    # print(dist, door_open)

    who_is_it("images/camera_0.jpg", database, FRmodel)
