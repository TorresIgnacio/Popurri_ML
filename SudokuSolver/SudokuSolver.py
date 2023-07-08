from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard, ModelCheckpoint
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
keras = tf.keras

BATCH_SIZE = 640
TEST_SIZE = 0.2
VAL_SIZE = 0.2
DATA_LEN = 100000
EPOCHS = 10
MODEL_PATH = "./ModelPrueba_"


class DataGenerator(keras.utils.Sequence):
    def __init__(self, df, batch_size=16, subset="train", shuffle=False, info={}):
        super().__init__()
        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.subset = subset
        self.info = info

        # self.data_path = path
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df)/self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        X = np.empty((self.batch_size, 9, 9, 1))
        y = np.empty((self.batch_size, 81, 1))
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        for i, f in enumerate(self.df['quizzes'].iloc[indexes]):
            self.info[index*self.batch_size+i] = f
            X[i,] = (np.array(list(map(int, list(f)))).reshape((9, 9, 1))/9)-0.5
        if self.subset == 'train':
            for i, f in enumerate(self.df['solutions'].iloc[indexes]):
                self.info[index*self.batch_size+i] = f
                y[i,] = np.array(list(map(int, list(f)))).reshape((81, 1)) - 1
        if self.subset == 'train':
            return X, y
        else:
            return X


sudoku_df = pd.read_csv('./sudoku.csv')
print(sudoku_df.head())
print(sudoku_df.info())

sudoku_digested = sudoku_df.head(DATA_LEN)

X = np.empty((len(sudoku_digested), 9, 9, 1))
y = np.empty((len(sudoku_digested), 81, 1))

for i, (quizz, solution) in enumerate(zip(sudoku_digested['quizzes'].values, sudoku_digested['solutions'].values)):
    X[i,] = (np.array([np.int32(val)
             for val in quizz]).reshape((9, 9, 1))/9)-0.5
    y[i,] = np.array([np.int32(val) for val in solution]).reshape((81, 1)) - 1

X_train_val, X_test, y_train_val, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=101)

X_train, X_val, y_train, y_val = train_test_split(X_train_val,
                                                  y_train_val,
                                                  test_size=0.2,
                                                  random_state=101)
train_idx = int(len(sudoku_digested)*(1-TEST_SIZE-VAL_SIZE))
val_idx = int(len(sudoku_digested)*(1-TEST_SIZE))
data = sudoku_digested.sample(frac=1).reset_index(drop=True)
training_generator = DataGenerator(
    data.iloc[:train_idx], subset="train", batch_size=BATCH_SIZE)
validation_generator = DataGenerator(
    data.iloc[train_idx:val_idx], subset="train",  batch_size=BATCH_SIZE)
test_generator = DataGenerator(
    data.iloc[val_idx:], subset='train', batch_size=BATCH_SIZE)


model = Sequential()
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Conv2D(filters=9,kernel_size=(3,3),strides=2,padding='same'))
# model.add(keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=3,padding='same'))
# model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
model.add(Dense(60, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Flatten())
model.add(Dense(81*9))
model.add(Reshape((-1, 9)))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
checkpoint2 = ModelCheckpoint(filepath=MODEL_PATH,
                              monitor='val_loss',
                              verbose=1,
                              save_best_only=True,
                              mode='min')
board = TensorBoard(log_dir='.\\logs\\fit',
                    histogram_freq=1,
                    write_graph=True,
                    write_images=True,
                    update_freq='epoch',
                    profile_batch=2,
                    embeddings_freq=1)

model.fit(x=training_generator,
          validation_data=validation_generator,
          epochs=EPOCHS,
          callbacks=[checkpoint2, board],
          verbose=1)

model.save(MODEL_PATH)
losses = pd.DataFrame(model.history.history)
losses.plot()
losses.to_csv('SudokuModelPruebaLosses.csv')
plt.show(block=False)
