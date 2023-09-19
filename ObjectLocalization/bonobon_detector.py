from imageio.v2 import imread
from keras.optimizers import Adam, SGD
from keras.layers import Flatten, Dense
from keras.models import Model
from skimage.transform import resize


import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras


CURRENT_DIR = os.path.dirname(__file__)
IMG_DIM = 200
NUM_BATCHES = 50
WEIGHTS_PATH = os.path.join(CURRENT_DIR, './models/bonobon/bonobon_weights')

bon = imread(os.path.join(CURRENT_DIR, './bonobon.png'))


for i in range(bon.shape[0]):
    for j in range(bon.shape[1]):
        if sum(bon[i, j, :3]) == 255*3:
            bon[i, j, :3] = 0

plt.imshow(bon)
# plt.show()
bon = np.array(bon)
bon_H, bon_W, _ = bon.shape


def image_generator(batch_size=64):
    while True:
        for _ in range(NUM_BATCHES):
            X = np.zeros((batch_size, IMG_DIM, IMG_DIM, 3))
            Y = np.zeros((batch_size, 4))

            for i in range(batch_size):
                scale = np.random.uniform(0.5, 1.5)
                new_height, new_width = int(scale * bon_H), int(scale * bon_W)
                row0 = np.random.randint(IMG_DIM - new_height)
                col0 = np.random.randint(IMG_DIM - new_width)
                row1 = row0 + new_height
                col1 = col0 + new_width
                X[i, row0:row1, col0:col1, :] = resize(
                    bon[:, :, :3], output_shape=(new_height, new_width, 3), preserve_range=True).astype(np.uint8)
                Y[i, 0] = row0/IMG_DIM
                Y[i, 1] = col0/IMG_DIM

                Y[i, 2] = (row1 - row0)/IMG_DIM  # Height
                Y[i, 3] = (col1 - col0)/IMG_DIM  # Width

            yield X / 255., Y


# p = next(image_generator())[0]
# for img in p:
#     plt.imshow(img)
#     plt.show(block=False)
#     plt.pause(1)

vgg = tf.keras.applications.VGG16(
    input_shape=[IMG_DIM, IMG_DIM, 3],
    include_top=False,
    weights='imagenet'
)
vgg.trainable = False

x = Flatten()(vgg.output)
x = Dense(4, activation='sigmoid')(x)
model = Model(vgg.input, x)

model.load_weights(WEIGHTS_PATH)
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))
# model.fit(image_generator(), steps_per_epoch=NUM_BATCHES, epochs=10)


def make_prediction():
    X = next(image_generator())[0]
    preds = model.predict(X)
    for i, pred in enumerate(preds):
        print(pred)
        fig, ax = plt.subplots(1)
        ax.imshow(X[i])
        rect = plt.Rectangle(
            (pred[1] * IMG_DIM, pred[0] * IMG_DIM),
            pred[3] * IMG_DIM, pred[2] * IMG_DIM,
            linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.show(block=False)
        plt.pause(2)
        plt.close()


make_prediction()
