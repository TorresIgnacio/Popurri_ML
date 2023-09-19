from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.layers import Flatten, Dense
import numpy as np
import tensorflow as tf
import os

from matplotlib import pyplot as plt

keras = tf.keras
NUM_BATCHES = 50
IMG_SIZE = 100
CURRENT_DIR = os.path.dirname(__file__)
WEIGHTS_PATH = os.path.join(CURRENT_DIR, './models/whitebox/whitebox_weights')


vgg = tf.keras.applications.VGG16(
    input_shape=[IMG_SIZE, IMG_SIZE, 3],
    include_top=False,
    weights='imagenet'
)
vgg.trainable = False

x = Flatten()(vgg.output)
x = Dense(4, activation='sigmoid')(x)
model = Model(vgg.input, x)


def image_generator(batch_size=64):
    while True:
        for _ in range(NUM_BATCHES):
            X = np.zeros((batch_size, IMG_SIZE, IMG_SIZE, 3))
            y = np.zeros((batch_size, 4))

            for i in range(batch_size):
                row0 = np.random.randint(IMG_SIZE - 10)
                col0 = np.random.randint(IMG_SIZE - 10)
                row1 = np.random.randint(row0, IMG_SIZE)
                col1 = np.random.randint(col0, IMG_SIZE)
                X[i, row0:row1, col0:col1, :] = 1
                y[i, 0] = row0/IMG_SIZE  # y0
                y[i, 1] = col0/IMG_SIZE  # x0
                y[i, 2] = (row1 - row0) / IMG_SIZE  # Height
                y[i, 3] = (col1 - col0) / IMG_SIZE  # Width

            yield X, y


model.load_weights(WEIGHTS_PATH)
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))

# model.fit_generator(
#     image_generator(),
#     steps_per_epoch=NUM_BATCHES,
#     epochs=5
# )


def make_prediction():
    x = np.zeros((IMG_SIZE, IMG_SIZE, 3))
    row0 = np.random.randint(IMG_SIZE - 10)
    col0 = np.random.randint(IMG_SIZE - 10)
    row1 = np.random.randint(row0, IMG_SIZE)
    col1 = np.random.randint(col0, IMG_SIZE)
    x[row0:row1, col0:col1] = 1
    print(row0, row1, col0, col1)

    X = np.expand_dims(x, 0)
    preds = model.predict(X)[0]
    print(preds)
    fig, ax = plt.subplots(1)
    ax.imshow(x)
    rect = plt.Rectangle(
        (preds[1] * IMG_SIZE, preds[0] * IMG_SIZE),
        preds[3] * IMG_SIZE, preds[2] * IMG_SIZE,
        linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)


while (True):
    make_prediction()
    plt.show(block=False)
    plt.pause(2)
    plt.close()
