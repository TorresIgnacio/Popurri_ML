from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from os import system
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
keras = tf.keras

BATCH_SIZE = 32
IMG_SIZE = 160
IMG_HEIGHT = IMG_SIZE
IMG_WIDTH = IMG_SIZE
AUTOTUNE = tf.data.AUTOTUNE

# Display 9 examples of the images with the prediction and true label side by side


def display_images(images, plot_count, predictions=[], labels=[], with_title=False, only_incorrect=False):
    for i in range(len(images)):
        if plot_count == 9:
            plt.figure(figsize=(10, 10))
            plot_count = 0
        if (only_incorrect and predictions[i] == labels[i]):
            continue
        ax = plt.subplot(3, 3, plot_count+1)
        image = images[i]
        plt.imshow(image)
        plt.axis("off")
        if with_title:
            plt.title(
                f"predicted={class_names[predictions[i]]} / real={class_names[labels[i]]}")
        plot_count += 1
    return plot_count


def display_all_images(dataset, test_preds):
    plot_count = 0
    batch_count = 0
    plt.figure(figsize=(10, 10))
    for images, labels in dataset:
        lower_lim = batch_count * BATCH_SIZE
        upper_lim = lower_lim + BATCH_SIZE
        plot_count = display_images(
            images.numpy().astype("uint8"),
            plot_count,
            predictions=test_preds[lower_lim:upper_lim],
            labels=labels.numpy(),
            with_title=True,
            only_incorrect=True)
        batch_count += 1
    plt.show()


def visualize_filters(model, img_size):
    conv_layer_index = [2, 4]
    outputs = [model.layers[i].output for i in conv_layer_index]
    model_short = keras.models.Model(inputs=model.inputs, outputs=outputs)

    from keras.utils import load_img, img_to_array
    img = load_img('./dataset/test_set/dogs/dog.4009.jpg',
                   target_size=(img_size, img_size))
    plt.figure()
    plt.imshow(img)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    feature_output = model_short.predict(img)

    columns = 8
    rows = 4
    for ftr in feature_output:
        fig = plt.figure(figsize=(12, 12))
        for i in range(1, columns*rows + 1):
            fig = plt.subplot(rows, columns, i)
            fig.set_xticks([])
            fig.set_yticks([])
            plt.imshow(ftr[0, :, :, i-1], cmap='gray')
    plt.show()


system('cls')
losses = [pd.read_csv(
    f"./DogClassifier/logs/KerasSolutionLosses{i}.csv") for i in range(1, 9)]
plt.figure(figsize=(15, 10))
for i in range(len(losses)):
    ax = plt.subplot(3, 3, i+1)
    losses[i].drop(['Unnamed: 0'], axis=1, inplace=True)
    # labels = losses[i].columns.values
    # ax.plot(losses[i], label=labels)
    ax.plot(losses[i]['loss'], label='loss')
    ax.plot(losses[i]['val_loss'], label='val_loss')
    plt.title(f"Model {i+1}")
    # ax.set_xlabel('EPOCHS')
    ax.set_ylabel('LOSS')
    ax.legend()

print(losses[2].info())
print(losses[2].head())

plt.xlabel(xlabel='EPOCHS')
plt.show()
# system('pause')


test_dir = "./DogClassifier/datasets/test_set"
test_ds = keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False
)
class_names = list(test_ds.class_names)
class_names.append('other')
print(class_names)

model = keras.saving.load_model(
    "./DogClassifier/Models/CNNSolutionModel8", compile=True)
model_mn2 = keras.models.Sequential()
model_mn2.add(keras.layers.Rescaling(scale=1./255, offset=-1))
model_mn2.add(keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3)))

preds_mn2 = model_mn2.predict(test_ds)
predicted_labels_mn2 = np.argmax(preds_mn2, axis=1)
# 281-285 = gato
y_pred_mn2 = [0 if pred >= 281 and pred <= 285 else (
    1 if pred >= 151 and pred <= 268 else 2) for pred in predicted_labels_mn2]


y_pred = model.predict(test_ds)
test_preds = [1 if prediction >= 0.5 else 0 for prediction in y_pred]
# test_preds = np.where(model.predict(test_ds) > 0.5, 1, 0)

y_test = [label.numpy() for (_, labels) in test_ds for label in labels]
print("----------Custom Model---------")
test_accuracy = accuracy_score(y_test, test_preds)
print(f'Test Accuracy: {test_accuracy:.4f}')
print(confusion_matrix(y_pred=test_preds, y_true=y_test))
print(classification_report(y_pred=test_preds, y_true=y_test))
print("----------MobileNetV2 Model---------")
test_accuracy = accuracy_score(y_test, y_pred_mn2)
print(f'Test Accuracy: {test_accuracy:.4f}')
print(confusion_matrix(y_pred=y_pred_mn2, y_true=y_test))
print(classification_report(y_pred=y_pred_mn2, y_true=y_test, labels=[0, 1]))

plt.show()
system('pause')
# display_all_images(dataset=test_ds, test_preds=y_pred_mn2)
display_all_images(dataset=test_ds, test_preds=test_preds)
