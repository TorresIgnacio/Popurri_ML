
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
keras = tf.keras
from os import system

system('cls')
BATCH_SIZE = 32
IMG_SIZE = 60
IMG_HEIGHT = IMG_SIZE
IMG_WIDTH = IMG_SIZE
AUTOTUNE = tf.data.AUTOTUNE

data_dir = "./dataset/training_set"
test_dir = "./dataset/test_set"

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

losses = pd.read_csv(f"./KerasSolutionLosses3.csv")
losses.drop(['Unnamed: 0'], axis=1, inplace=True)
last_epoch = len(losses)
print(last_epoch)
model = keras.saving.load_model(".\CNNSolutionModel3", compile=True)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x=train_ds, validation_data=val_ds, epochs=last_epoch+100, verbose=1, initial_epoch=last_epoch)
model.save("CNNSolutionModel3")
losses = pd.concat([losses, pd.DataFrame(model.history.history)])
losses.to_csv("KerasSolutionLosses3.csv")
plt.plot(losses)
plt.show()