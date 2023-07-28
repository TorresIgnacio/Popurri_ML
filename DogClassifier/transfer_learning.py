import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import system
layers = tf.keras.layers


system('cls')
BATCH_SIZE = 32
IMG_SIZE = (160, 160)
AUTOTUNE = tf.data.AUTOTUNE

train_dir = './dataset/training_set'
test_dir = './dataset/test_set'

train_dataset, val_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                            shuffle=True,
                                            validation_split=0.2,
                                            seed=120,
                                            subset='both',
                                            label_mode='binary',
                                            batch_size=BATCH_SIZE,
                                            image_size=IMG_SIZE)

test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                            shuffle=False,
                                            label_mode='binary',
                                            batch_size=BATCH_SIZE,
                                            image_size=IMG_SIZE)

class_names = train_dataset.class_names


train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip('horizontal'),
  layers.RandomRotation(0.1),
  layers.RandomBrightness(0.2),
  layers.RandomContrast(0.3),
  layers.RandomZoom(0.4)
])

rescale = layers.Rescaling(1./127.5, offset=-1)

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  first_image = images[0]
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
    plt.imshow(augmented_image[0]/255)
    plt.axis("off")

IMG_SHAPE = IMG_SIZE + (3,)
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)
base_model.trainable = False

base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

history = model.fit(train_dataset,
                    epochs=50,
                    validation_data=val_dataset)
model.save("CNNSolutionModel8")
losses = pd.DataFrame(history.history)
losses.to_csv("KerasSolutionLosses8.csv")
y_pred = model.predict(test_dataset)
predictions = [(1 if pred > 0.5 else 0) for pred in y_pred]
#y_test = [label.numpy() for (_, labels) in test_ds for label in labels]
y_test = [label for (_, labels) in test_dataset for label in labels]

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_pred=predictions, y_true=y_test))
print(classification_report(y_pred=predictions, y_true=y_test))

plt.figure()
plt.plot(history.history['loss'], color='red', label='Training Loss')
plt.plot(history.history['val_loss'], color='blue', label='Validation Loss')
plt.legend()
plt.title("losses")
plt.show()
            