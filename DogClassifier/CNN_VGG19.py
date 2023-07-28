import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from os import system
keras = tf.keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


system('cls')
print(tf.__version__)
# Check if a GPU is available
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

 
IMG_SIZE = 224
BATCH_SIZE = 32
IMG_HEIGHT = IMG_SIZE
IMG_WIDTH = IMG_SIZE
AUTOTUNE = tf.data.AUTOTUNE

data_dir = "./dataset/training_set"
test_dir = "./dataset/test_set"


# Preprocessing the Training set
#train_datagen = ImageDataGenerator(rescale = 1./255,
                                   #shear_range = 0.2,
                                   #zoom_range = 0.2,
                                   #brightness_range=(-1.0, 1.0),
                                   #validation_split=0.2,
                                   #horizontal_flip = True)
#train_ds = train_datagen.flow_from_directory('dataset/training_set',
                                                 #target_size = (IMG_HEIGHT, IMG_WIDTH),
                                                 #batch_size = 32,
                                                 #subset="training",
                                                 #class_mode = 'binary')
#val_ds = train_datagen.flow_from_directory('dataset/training_set',
                                                 #target_size = (IMG_HEIGHT, IMG_WIDTH),
                                                 #batch_size = 32,
                                                 #subset="validation",
                                                 #class_mode = 'binary')

## Preprocessing the Test set
#test_datagen = ImageDataGenerator(rescale = 1./255)
#test_ds = test_datagen.flow_from_directory('dataset/test_set',
                                            #batch_size = 32,
                                            #target_size = (IMG_HEIGHT, IMG_WIDTH),
                                            #class_mode = 'binary')

train_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=71,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=71,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

test_ds = keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = list(train_ds.class_names)
print(class_names)

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2),
  ]
)

cnn = keras.models.Sequential([
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(4096, activation='relu'),
    layers.Dense(4096, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=5)
cnn.compile(optimizer='adam', loss='binary_crossentropy')

print(cnn.summary())
cnn.fit(x=train_ds, validation_data=val_ds, epochs=30, callbacks=[early_stop], verbose=1)
cnn.save("VGG19_model")
losses = pd.DataFrame(cnn.history.history)
losses.plot()
losses.to_csv("VGG19_losses.csv")
y_pred = cnn.predict(test_ds)
predictions = [(1 if pred > 0.5 else 0) for pred in y_pred]
#y_test = [label.numpy() for (_, labels) in test_ds for label in labels]
y_test = [label for (_, labels) in test_ds for label in labels]

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_pred=predictions, y_true=y_test))
print(classification_report(y_pred=predictions, y_true=y_test))
plt.show()