import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import io
from PIL import Image

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler, ModelCheckpoint, CSVLogger, ReduceLROnPlateau

import os
import numpy as np
import math
import datetime
import pandas as pd 

print("Version: ", tf.__version__)
tf.get_logger().setLevel("INFO")

splits, info = tfds.load('horses_or_humans', as_supervised=True, with_info=True, split=['train[:80%]', 'train[:80%]', 'test'])

(train_examples, validation_examples, test_examples) = splits

num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes

SIZE = 150
IMAGE_SIZE = (SIZE, SIZE)

def formart_image(image, label):
    image = tf.image.resize(image, IMAGE_SIZE) / 255.0
    return image, label


BATCH_SIZE = 32

train_batches = train_examples.shuffle(num_examples // 4).map(formart_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.map(formart_image).batch(BATCH_SIZE).prefetch(1)
test_batches = test_examples.map(formart_image).batch(BATCH_SIZE).prefetch(1)

for image_batch, label_batch, in train_batches.take(1):
    pass

image_batch.shape

def build_model(dense_units, input_shape=IMAGE_SIZE+(3,)):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(dense_units, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    return model

#tensorboard

!rm -rf logs 

model=build_model(dense_units=256)
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir)

model.fit(train_batches, epochs=10, validation_data=validation_batches, callbacks=[tensorboard_callback])

%tensorboard --logdir logs

#model Checkpoint

model=build_model(dense_units=256)
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_batches, epochs=5, validation_data=validation_batches, callbacks=[ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.h5', verbose=1)])


#earlyStopping

model=build_model(dense_units=256)
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model.fit(train_batches, epochs=10, validation_data=validation_batches, verbose=2, callbacks=[EarlyStopping(
    patience=3,
    min_delta=0.05,
    baseline=0.8,
    mode='min',
    monitor='val_loss',
    restore_best_weights=True,
    verbose=1
)])


#CSV Logger

model=build_model(dense_units=256)
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

csv_file = "training.csv"

model.fit(train_batches, epochs=10, validation_data=validation_batches, callbacks=[CSVLogger(csv_file)])


# class Callback(object):
#     def __init__(self):
#         self.validation_data = None
#         self.model = None
    
#     def on_epoch_begin(self, epoch, logs=None):
#         ""
    
#     def on_epoch_end(self, epoch, logs=None):
#         ""

#     def on_(train|test|predict)_begin(self, logs=None):
#         ""
    
#     def on_(train|test|predict)_end(self, logs=None):
#         ""

#     def on_(train|test|predict)_batch_begin(self, logs=None):
#         ""
    
#     def on_(train|test|predict)_batch_end(self, logs=None):
#         ""