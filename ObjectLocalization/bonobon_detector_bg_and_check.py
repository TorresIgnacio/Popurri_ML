from imageio.v2 import imread
from keras.optimizers import Adam, SGD
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.losses import binary_crossentropy
from skimage.transform import resize


import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras


CURRENT_DIR = os.path.dirname(__file__)
IMG_DIM = 200
NUM_BATCHES = 50
WEIGHTS_PATH = os.path.join(
    CURRENT_DIR, './models/bonobon_with_bg_and_check/bonobon_bg_and_check_weights')

bon = imread(os.path.join(CURRENT_DIR, './bonobon.png'))
bg = imread(os.path.join(CURRENT_DIR, './background.png'))


# for i in range(bon.shape[0]):
#     for j in range(bon.shape[1]):
#         if sum(bon[i, j, :3]) == 255*3:
#             bon[i, j, :3] = 0

for i in range(bon.shape[0]):
    for j in range(bon.shape[1]):
        if sum(bon[i, j, :3]) == 255*3:
            bon[i, j, 3] = 0

plt.imshow(bon)
# plt.show()
# bon = np.array(bon)
bon_H, bon_W, _ = bon.shape


def image_generator(batch_size=64):
    while True:
        for _ in range(NUM_BATCHES):
            X = np.zeros((batch_size, IMG_DIM, IMG_DIM, 3), dtype=np.uint8)
            Y = np.zeros((batch_size, 5))

            for i in range(batch_size):
                # select bg
                bg_h, bg_w, _ = bg.shape
                rnd_h = np.random.randint(bg_h - IMG_DIM)
                rnd_w = np.random.randint(bg_w - IMG_DIM)
                X[i] = bg[rnd_h:rnd_h+IMG_DIM, rnd_w:rnd_w +
                          IMG_DIM, :3].copy().astype(np.uint8)

                appear = (np.random.random() > 0.5)
                if (appear):
                    # scale
                    scale = np.random.uniform(0.5, 1.5)
                    new_height, new_width = int(
                        scale * bon_H), int(scale * bon_W)

                    # choose location
                    row0 = np.random.randint(IMG_DIM - new_height)
                    col0 = np.random.randint(IMG_DIM - new_width)
                    row1 = row0 + new_height
                    col1 = col0 + new_width

                    # resize
                    obj = resize(
                        bon, output_shape=(new_height, new_width), preserve_range=True).astype(np.uint8)

                    # flip
                    if np.random.random() < 0.5:
                        obj = np.fliplr(obj)

                    # add object to background
                    mask = (obj[:, :, 3] == 0)
                    bg_slice = X[i, row0:row1, col0:col1, :]
                    bg_slice = np.expand_dims(
                        mask, -1) * bg_slice  # (h,w,1)*(h,w,3)
                    bg_slice += obj[:, :, :3]
                    X[i, row0:row1, col0:col1, :] = bg_slice

                    Y[i, 0] = row0/IMG_DIM
                    Y[i, 1] = col0/IMG_DIM

                    Y[i, 2] = (row1 - row0)/IMG_DIM  # Height
                    Y[i, 3] = (col1 - col0)/IMG_DIM  # Width
                Y[i, 4] = appear

            yield (X / 255., Y)


# p = next(image_generator())[0]
# for img in p:
#     plt.imshow(img)
#     plt.show(block=False)
#     plt.pause(1)

# Create Model

def custom_loss(y_true, y_pred):
    # weights to choose what error is more important
    alpha = 2
    beta = 0.5

    bce = binary_crossentropy(y_true[:, :-1], y_pred[:, :-1])
    bce2 = binary_crossentropy(y_true[:, -1], y_pred[:, -1])

    # the bounding box error should not contribute to the loss if the object does not appears
    return alpha * bce * y_true[:, -1] + beta * bce2


vgg = tf.keras.applications.VGG16(
    input_shape=[IMG_DIM, IMG_DIM, 3],
    include_top=False,
    weights='imagenet'
)
vgg.trainable = False

x = Flatten()(vgg.output)
x = Dense(5, activation='sigmoid')(x)
model = Model(vgg.input, x)

model.load_weights(WEIGHTS_PATH)
model.compile(loss=custom_loss, optimizer=Adam(learning_rate=0.001))
# model.fit(image_generator(), steps_per_epoch=NUM_BATCHES, epochs=10)


def make_prediction():
    X = next(image_generator())[0]
    preds = model.predict(X)
    for i, pred in enumerate(preds):
        fig, ax = plt.subplots(1)
        ax.imshow(X[i])
        if pred[4] > 0.5:
            rect = plt.Rectangle(
                (pred[1] * IMG_DIM, pred[0] * IMG_DIM),
                pred[3] * IMG_DIM, pred[2] * IMG_DIM,
                linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.set_title(pred[:4])
        else:
            ax.set_title('No object')
        plt.show(block=False)
        plt.pause(2)
        plt.close()


make_prediction()
