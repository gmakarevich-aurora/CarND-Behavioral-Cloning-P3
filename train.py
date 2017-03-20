import json

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.visualize_util import plot
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import model as model_builder
import data

UDACITY_TRAINING_DATA_PATH = './data/driving_log.csv'
UDACITY_TRAINING_DATA_IMG_DIR = './data/IMG'

TRACK1_TRAINING_DATA_PATH = './track1_data/driving_log.csv'
TRACK1_TRAINING_DATA_IMG_DIR = './track1_data/IMG'

TRACK2_TRAINING_DATA_PATH = './track2_data/driving_log.csv'
TRACK2_TRAINING_DATA_IMG_DIR = './track2_data/IMG'

TRACK21_TRAINING_DATA_PATH = './track21_data/driving_log.csv'
TRACK21_TRAINING_DATA_IMG_DIR = './track21_data/IMG'

TRAINING_DATA_SOURCES = [
    (UDACITY_TRAINING_DATA_PATH, UDACITY_TRAINING_DATA_IMG_DIR),
    (TRACK1_TRAINING_DATA_PATH, TRACK1_TRAINING_DATA_IMG_DIR),
    (TRACK2_TRAINING_DATA_PATH, TRACK2_TRAINING_DATA_IMG_DIR),
    (TRACK21_TRAINING_DATA_PATH, TRACK21_TRAINING_DATA_IMG_DIR)
]

NB_EPOCH = 1000
PATIENCE = 15


if __name__ == '__main__':
    # Read the data from multiple runs.
    samples = data.read_samples_from_multiple_sources(
            TRAINING_DATA_SOURCES)
    # Split samples into training and validation set.
    train_samples, validation_samples = data.split_samples(
            samples, 0.25)

    # Build the generators, which may be used with fit_generator below.
    train_gen = data.ImageGenerator(train_samples)
    validation_gen = data.ImageGenerator(validation_samples)

    # Build the model. The parameter is the shape of the input layer.
    model = model_builder.build_model(
        data.get_model_input_shape(samples))

    # Log the summary of the model 
    model.summary()
    # Plot the architecture of the model.
    plot(model, to_file='model.png', show_shapes=True)

    # Compile the model.
    # The loss function is mean squared error
    # We use Adam optimizer, with learning rate 1e-4
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')

    # Persist trained model
    model_json = model.to_json()
    with open('model.json', 'w') as f:
        json.dump(model_json, f)

    # For iterative approach we load previously saved model.
    model.load_weights('model.h5')

    checkpoint = ModelCheckpoint(
            'model.h5', monitor='val_loss',
            verbose=1, save_best_only=True,
            save_weights_only=False, mode='auto')

    early_stopping = EarlyStopping(
            monitor='val_loss', min_delta=0,
            patience=PATIENCE, verbose=1, mode='auto')

    # Train the model.
    history_object = model.fit_generator(
            train_gen.generator(),
            samples_per_epoch=train_gen.total_images(),
            validation_data=validation_gen.generator(),
            nb_val_samples=validation_gen.total_images(),
            nb_epoch=NB_EPOCH,
            callbacks=[checkpoint, early_stopping])

    # summarize history for loss
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
