import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from os import system
keras = tf.keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LDALayer(tf.keras.layers.Layer):
    def __init__(self, num_classes, **kwargs):
        super(LDALayer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.lda = LinearDiscriminantAnalysis(n_components=num_classes - 1)
        
    def call(self, inputs):
        # Fit the LDA model to the input features and target values
        y = np.zeros_like(inputs[:, 0])  # Just a placeholder since LDA requires target variable
        self.lda.fit(inputs, y)
        
        # Transform the input features using the LDA projection
        projected_inputs = self.lda.transform(inputs)
        
        # Apply the weights and biases to the projected inputs
        outputs = self.add_weight(name='W', shape=(projected_inputs.shape[1], self.num_classes),
                                  initializer='glorot_uniform', trainable=True)(projected_inputs)
        outputs = tf.nn.bias_add(outputs, self.add_weight(name='b', shape=(self.num_classes,),
                                                          initializer='zeros', trainable=True))
        
        return outputs

system('cls')
print(tf.__version__)
# Check if a GPU is available
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

 
BATCH_SIZE = 128
IMG_SIZE = 60
IMG_HEIGHT = IMG_SIZE
IMG_WIDTH = IMG_SIZE
AUTOTUNE = tf.data.AUTOTUNE

data_dir = "./dataset/training_set"
test_dir = "./dataset/test_set"

# Part 1 - Data Preprocessing

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

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="both",
    seed=71,
    label_mode='binary',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

#val_ds = keras.utils.image_dataset_from_directory(
    #data_dir,
    #validation_split=0.2,
    #subset="validation",
    #seed=71,
    #label_mode='binary',
    #image_size=(IMG_HEIGHT, IMG_WIDTH),
    #batch_size=BATCH_SIZE
#)

test_ds = keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False
)

class_names = list(train_ds.class_names)
print(class_names)
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2)
    #layers.RandomBrightness(0.1)
  ]
)
data_augmentation2 = keras.Sequential(
  [
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
  ]
)

cnn = keras.models.Sequential([
    data_augmentation,
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPool2D(pool_size=4, strides=4),
    layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
    layers.MaxPool2D(pool_size=2, strides=2),
    #layers.Dropout(0.2),
    layers.Flatten(),
    #LDALayer(num_classes=128),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=30)
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(cnn.summary())
cnn.fit(x=train_ds, validation_data=val_ds, epochs=100, verbose=1)
cnn.save("CNNSolutionModel8")
losses = pd.DataFrame(cnn.history.history)
losses.plot()
losses.to_csv("KerasSolutionLosses8.csv")
y_pred = cnn.predict(test_ds)
predictions = [(1 if pred > 0.5 else 0) for pred in y_pred]
#y_test = [label.numpy() for (_, labels) in test_ds for label in labels]
y_test = [label for (_, labels) in test_ds for label in labels]

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_pred=predictions, y_true=y_test))
print(classification_report(y_pred=predictions, y_true=y_test))
plt.show()