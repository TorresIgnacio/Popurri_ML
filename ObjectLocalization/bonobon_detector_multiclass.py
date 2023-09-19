from imageio.v2 import imread
from keras.optimizers import Adam, SGD
from keras.layers import Concatenate, Flatten, Dense
from keras.models import Model
from keras.losses import binary_crossentropy, categorical_crossentropy
from keras.metrics import categorical_accuracy, binary_accuracy, mean_squared_error
from skimage.transform import resize


import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras


CURRENT_DIR = os.path.dirname(__file__)
IMG_DIM = 400
OBJ_MAX_DIM = 150
NUM_BATCHES = 50
WEIGHTS_PATH = os.path.join(
    CURRENT_DIR, './models/bonobon_multiclass/bonobon_multiclass_weights_4')

# bon = imread(os.path.join(CURRENT_DIR, './bonobon.png'))
bg = imread(os.path.join(CURRENT_DIR, './background.png'))

objects = []
objects.append(imread(os.path.join(CURRENT_DIR, './bon_negro.png')))
objects.append(imread(os.path.join(CURRENT_DIR, './bon_blanco.png')))
objects.append(imread(os.path.join(CURRENT_DIR, './caja_bon.png')))
objects.append(imread(os.path.join(CURRENT_DIR, './caja_bon_aerea.png')))

class_names = ['Bonobon Negro', 'Bonobon Blanco', 'Caja de Bonobon']
class_idx = {0: 0, 1: 1, 2: 2, 3: 2}

for i, obj in enumerate(objects):
    height, width, _ = obj.shape
    plt.show()
    transparency_layer = (obj[:, :, -1] // 255).astype(np.uint8)
    obj[:, :, :-1] *= transparency_layer[:, :, np.newaxis]
    if width > OBJ_MAX_DIM:
        new_height = (OBJ_MAX_DIM/width) * height
        objects[i] = resize(obj, output_shape=(
            new_height, OBJ_MAX_DIM), preserve_range=True).astype(np.uint8)


def image_generator(batch_size=64):
    while True:
        for _ in range(NUM_BATCHES):
            X = np.zeros((batch_size, IMG_DIM, IMG_DIM, 3), dtype=np.uint8)
            Y = np.zeros((batch_size, 8))

            for i in range(batch_size):
                # select bg
                bg_h, bg_w, _ = bg.shape
                rnd_h = np.random.randint(bg_h - IMG_DIM)
                rnd_w = np.random.randint(bg_w - IMG_DIM)
                X[i] = bg[rnd_h:rnd_h+IMG_DIM, rnd_w:rnd_w +
                          IMG_DIM, :3].copy().astype(np.uint8)

                appear = (np.random.random() > 0.25)
                if (appear):
                    # select object
                    bon = np.random.choice(objects)
                    bon_H, bon_W, _ = bon.shape
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

                    # location
                    Y[i, 0] = row0/IMG_DIM
                    Y[i, 1] = col0/IMG_DIM
                    Y[i, 2] = (row1 - row0)/IMG_DIM  # Height
                    Y[i, 3] = (col1 - col0)/IMG_DIM  # Width

                    # class

                    Y[i, 4 + class_idx[obj_idx]] = 1
                Y[i, 7] = appear

            yield (X / 255., Y)


# Create Model

def custom_loss(y_true, y_pred):
    # Targets = (row, col, depth, width, class1, class2, class3, object_appeared)
    bce = binary_crossentropy(y_true[:, :4], y_pred[:, :4])  # location
    cce = categorical_crossentropy(
        y_true[:, 4:7], y_pred[:, 4:7])  # object class
    bce2 = binary_crossentropy(y_true[:, -1], y_pred[:, -1])  # object appeared

    # the bounding box error should not contribute to the loss if the object does not appears
    return bce * y_true[:, -1] + 0.5 * cce * y_true[:, -1] + 0.25 * bce2


vgg = tf.keras.applications.VGG16(
    input_shape=[IMG_DIM, IMG_DIM, 3],
    include_top=False,
    weights='imagenet'
)
vgg.trainable = False

x = Flatten()(vgg.output)
x1 = Dense(4, activation='sigmoid')(x)  # Location
x2 = Dense(3, activation='softmax')(x)  # Object Class
x3 = Dense(1, activation='sigmoid')(x)  # Object Appeared
x = Concatenate()([x1, x2, x3])
model = Model(vgg.input, x)

model.load_weights(WEIGHTS_PATH)
model.compile(loss=custom_loss, optimizer=Adam(learning_rate=0.0001))
# model.fit(image_generator(), steps_per_epoch=NUM_BATCHES, epochs=10)


def plot_prediction(img, pred, y_true=None):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    if pred[7] > 0.5:
        rect = plt.Rectangle(
            (pred[1] * IMG_DIM, pred[0] * IMG_DIM),
            pred[3] * IMG_DIM, pred[2] * IMG_DIM,
            linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        if y_true is not None:
            rect_2 = plt.Rectangle(
                (y_true[1] * IMG_DIM, y_true[0] * IMG_DIM),
                y_true[3] * IMG_DIM, y_true[2] * IMG_DIM,
                linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect_2)
        class_pred_idx = np.argmax(pred[4:7])
        class_pred = class_names[class_pred_idx]
        ax.set_title(class_pred)
    else:
        ax.set_title('Producto no detectado')


def make_prediction():
    X = next(image_generator())[0]
    preds = model.predict(X)
    for i, pred in enumerate(preds):
        plot_prediction(img=X[i], pred=pred)
        plt.show(block=False)
        plt.pause(2)
        # input()
        plt.close()


def predict_one_image(img: np.ndarray):
    test = resize(img, output_shape=(IMG_DIM, IMG_DIM),
                  preserve_range=True).astype(np.uint8)
    test = test[:, :, :3] / 255.
    test = np.expand_dims(test, 0)
    pred = model.predict(test)[0]
    plot_prediction(img=test[0], pred=pred)
    plt.show()


# make_prediction()

def make_analysis(y_true, y_pred):

    # binary accuracy
    appeared_acc = binary_accuracy(
        y_true=y_true[:, 7], y_pred=(y_pred[:, 7] > 0.5)*1)
    print(f'binary accuracy = {appeared_acc * 100}%')

    # class accuracy
    appeared_idx = np.argwhere((y_true[:, 7] > 0.5) & (y_pred[:, 7] > 0.5))
    appeared_idx = np.reshape(appeared_idx, newshape=(appeared_idx.shape[0]))
    y_true = y_true[appeared_idx]
    y_pred = y_pred[appeared_idx]
    class_acc = categorical_accuracy(
        y_true=y_true[:, 4:7], y_pred=y_pred[:, 4:7])
    class_acc = sum(class_acc)/len(class_acc)
    print(f'class classification accuracy = {class_acc * 100}%')

    # localization accuracy
    localization_error = []
    for i in range(4):
        localization_error.append(mean_squared_error(
            y_true=y_true[:, i]*IMG_DIM, y_pred=y_pred[:, i]*IMG_DIM))
    localization_error = np.sqrt(localization_error)
    print(f'x error = {localization_error[1]}')
    print(f'y error = {localization_error[0]}')
    print(f'width error = {localization_error[3]}')
    print(f'height error = {localization_error[2]}')

    avg_localization_error = sum(localization_error)/len(localization_error)
    print(
        f'average localization error = {avg_localization_error}')


def compare_models(X, y_true, last_model=5):
    for i in range(2, last_model):
        weights_path = os.path.join(
            CURRENT_DIR, f'./models/bonobon_multiclass/bonobon_multiclass_weights_{i}')
        model.load_weights(weights_path)
        y_pred = model.predict(X)
        print(f'\n\nModel {i} results')
        make_analysis(y_true=y_true, y_pred=y_pred)
