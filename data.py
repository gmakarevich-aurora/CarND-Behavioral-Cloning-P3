import os
import csv
import cv2
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

BATCH_SIZE = 128
FLIP_STEERING_MIN = 0.3
STRAIGTH_STEERING_LIMIT = 0.2
STRAIGTH_STEERING_KEEP_PROB = 0.3

class Sample(object):
    def __init__(self, image_path, steering, needs_flip=False):
        self.image_path = image_path
        self.steering = steering
        self.needs_flip = needs_flip

    def get_image_path(self):
        return self.image_path

    def get_steering(self):
        return self.steering

    def get_needs_flip(self):
        return self.needs_flip

def read_samples_from_multiple_sources(sources):
    samples = []
    for source in sources:
        (log_file, img_dir) = source
        samples.extend(read_samples(log_file, img_dir))

    return shuffle(samples)

def read_samples(log_file, img_dir):
    samples = []
    with open(log_file, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # Skip header
        next(reader)
        for line in reader:
            steering = float(line[3])
            if abs(steering) < STRAIGTH_STEERING_LIMIT:
                v = random.random()
                if v > STRAIGTH_STEERING_KEEP_PROB:
                    continue
            center_img = '%s/%s' % (
                img_dir, line[0].split('/')[-1])
            left_img = '%s/%s' % (
                img_dir, line[1].split('/')[-1])
            right_img = '%s/%s' % (
                img_dir, line[2].split('/')[-1])
            samples.extend([
                Sample(center_img, steering),
                Sample(left_img, steering + 0.25),
                Sample(right_img, steering - 0.25)
            ])
            if (abs(steering) > FLIP_STEERING_MIN):
                samples.extend([
                    Sample(center_img, steering, needs_flip=True),
                    Sample(left_img, steering + 0.25, needs_flip=True),
                    Sample(right_img, steering - 0.25, needs_flip=True)
                ])


    return shuffle(samples)


def split_samples(samples, test_size):
    return train_test_split(samples, test_size=test_size)


def get_model_input_shape(samples):
    image_path = samples[0].get_image_path()
    image = cv2.imread(image_path)
    image = process_image(image)
    return image.shape


def process_image(img):
    image = img[50:140, :, :]
    image = cv2.GaussianBlur(image, (3,3), 0)
    image = cv2.resize(image, (200, 66), interpolation = cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    return image


class ImageGenerator(object):

    def __init__(self, samples, batch_size=BATCH_SIZE):
        self.samples = samples
        self.batch_size = batch_size

    def total_images(self):
        return len(self.samples)

    def generator(self):
        num_samples = len(self.samples)
        while 1: # Loop forever so the generator never terminates
            shuffle(self.samples)
            for offset in range(0, num_samples, self.batch_size):
                batch_samples = self.samples[offset:offset+self.batch_size]
                images = []
                angles = []
                for batch_sample in batch_samples:
                    image = process_image(
                        cv2.imread(batch_sample.get_image_path()))
                    steering = batch_sample.get_steering()
                    if batch_sample.get_needs_flip():
                        steering *= - 1
                        image = cv2.flip(image, 1)
                    images.extend([image])
                    angles.extend([steering])

                X_train = np.array(images)
                y_train = np.array(angles)
                yield shuffle(X_train, y_train)

