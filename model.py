from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.regularizers import l2, activity_l2
from keras.layers.advanced_activations import ELU

def build_model(input_shape):
    # We are building a model from
    # "End to End Learning for Self-Driving Cars"
    # Nvidia's paper.
    print (input_shape)
    model = Sequential()

    model.add(Lambda(lambda x: (x / 127.5) - 1.0, input_shape=input_shape))

    model.add(Convolution2D(
        24, 5, 5, border_mode='valid',
        W_regularizer=l2(0.001),
        subsample=(2, 2),
        input_shape=input_shape))
    model.add(ELU())
    model.add(Convolution2D(
        36, 5, 5, border_mode='valid',
        W_regularizer=l2(0.001),
        subsample=(2, 2)))
    model.add(ELU())
    model.add(Convolution2D(
        48, 5, 5, border_mode='valid',
        W_regularizer=l2(0.001),
        subsample=(2, 2)))
    model.add(ELU())
    model.add(Convolution2D(
        64, 3, 3, border_mode='valid',
        W_regularizer=l2(0.001),
        subsample=(1, 1)))
    model.add(ELU())
    model.add(Convolution2D(
        64, 3, 3, border_mode='valid',
        W_regularizer=l2(0.001),
        subsample=(1, 1)))
    model.add(ELU())

    model.add(Flatten())
    model.add(Dense(100, W_regularizer=l2(0.001)))
    model.add(ELU())

    model.add(Dense(50, W_regularizer=l2(0.001)))
    model.add(ELU())

    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(ELU())

    model.add(Dense(1))

    return model

