import seaborn as sns
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import copy
from multiprocessing import Pool
keras = tf.keras

BATCH_SIZE = 640
TEST_SIZE = 0.2
VAL_SIZE = 0.2
DATA_LEN = 1000000
NUMBER_OF_TESTS = 1
NUMBER_OF_BATCHES = 1
MODEL_PATH = "./SudokuSolverMostAccurateModel"


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


def norm(a):
    return (a/9)-.5


def denorm(a):
    return (a+.5)*9


def inference_sudoku(sample, debug=False):
    '''
        This function solve the sudoku by filling blank positions one by one.
    '''
    start_time = time.perf_counter()
    PROB_THRESHOLD = 0.991
    feat = copy.copy(sample)
    model = keras.saving.load_model(MODEL_PATH, compile=True)
    while (1):
        out = model.predict(feat.reshape((1, 9, 9, 1)), verbose=0)
        out = out.squeeze()
        pred = np.argmax(out, axis=1).reshape((9, 9))+1
        prob = np.max(out, axis=1).reshape((9, 9))
        feat = denorm(feat).reshape((9, 9))
        mask = (feat == 0)

        if (mask.sum() == 0):
            break
        prob_new = prob*mask
        ind = np.argwhere(prob_new >= PROB_THRESHOLD)
        if (len(ind) == 0):
            ind = np.argmax(prob_new)
            x, y = (ind//9), (ind % 9)
            feat[x, y] = pred[x, y]
        else:
            for x, y in ind:
                feat[x, y] = pred[x, y]
        feat = norm(feat)
    end_time = time.perf_counter()
    if debug:
        print(f'prediction time: {(end_time - start_time):.2f}s')
    return pred


def display_sudoku(pred, sol, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    fig.suptitle(title)
    ax1.set_title("Prediction")
    ax2.set_title("Solution")
    sns.heatmap(pred, cbar=False, linewidths=1,
                linecolor='black', annot=True, ax=ax1)
    ax1.pcolormesh(
        np.arange(pred.shape[0]+1), np.arange(pred.shape[1]+1), pred != sol, alpha=0.5)
    sns.heatmap(sol, cbar=False, linewidths=1,
                linecolor='black', annot=True, ax=ax2)
    ax2.pcolormesh(
        np.arange(sol.shape[0]+1), np.arange(sol.shape[1]+1), sol != pred, alpha=0.5)
    ax1.grid(which='major', color='green', linewidth=2)
    ax2.grid(which='major', color='green', linewidth=2)
    ax1.set_xticks([3, 6])
    ax1.set_yticks([3, 6])
    ax2.set_xticks([3, 6])
    ax2.set_yticks([3, 6])
    plt.show(block=False)


def eval_score(y_pred, y_true, title):
    # print(f"""
    # \n{"EJEMPLO".center(50,"-")}
    # \n{y_pred[1,]}
    # \n{"-".center(50,"-")}
    # \n{y_true[1,]}
    # \n{"-".center(50,"-")}
    # """)
    display_sudoku(y_pred[0,], y_true[0,], title)
    accuracy_score = np.sum(y_pred == y_true)/(81*len(y_pred))
    results = y_pred == y_true
    correct_sudokus = sum(map(lambda x: np.all(x), results))
    print(f'accuracy = {accuracy_score}')
    print(
        f'correct sudokus = {correct_sudokus} of {len(y_pred)} ({correct_sudokus/len(y_pred):.2f})')


def main():
    sudoku_df = pd.read_csv('./sudoku.csv')
    print(sudoku_df.head())
    print(sudoku_df.info())

    sudoku_digested = sudoku_df.head(DATA_LEN)

    X = np.empty((len(sudoku_digested), 9, 9, 1))
    y = np.empty((len(sudoku_digested), 81, 1))

    for i, (quizz, solution) in enumerate(zip(sudoku_digested['quizzes'].values, sudoku_digested['solutions'].values)):
        X[i,] = (np.array([np.int32(val)
                 for val in quizz]).reshape((9, 9, 1))/9)-0.5
        y[i,] = np.array([np.int32(val)
                         for val in solution]).reshape((81, 1)) - 1

    val_idx = int(len(sudoku_digested)*(1-TEST_SIZE))
    data = sudoku_digested.sample(frac=1).reset_index(drop=True)
    test_generator = DataGenerator(
        data.iloc[val_idx:], subset='train', batch_size=BATCH_SIZE)

    model = keras.saving.load_model(MODEL_PATH, compile=True)

    predictions = model.predict(test_generator)
    y_pred = np.zeros((len(predictions), 9, 9))
    y_solutions = np.zeros((len(predictions), 9, 9))
    for sudoku_number, pred in enumerate(predictions):
        for i, cell in enumerate(pred):
            y_pred[sudoku_number, np.int32(i/9), i % 9] = np.argmax(cell)+1

    for i in range(int(len(predictions)/BATCH_SIZE)):
        _, solutions = test_generator.__getitem__(i)
        for j, solution in enumerate(solutions):
            solution = solution.reshape(9, 9) + 1
            solution_number = i*BATCH_SIZE + j
            y_solutions[solution_number,] = solution

    title1 = "FIRST APPROACH - ALL PREDICTIONS AT ONCE"
    print(title1)
    eval_score(y_pred=y_pred, y_true=y_solutions, title=title1)

    # Second approach
    # Try over 100 tests because trying over the whole test set takes 40 hours!
    title2 = "SECOND APPROACH - ONE PREDICTION AT A TIME"
    print(title2)
    start_time = time.perf_counter()
    y_pred2 = np.zeros((NUMBER_OF_BATCHES * NUMBER_OF_TESTS, 9, 9))
    for batch in range(NUMBER_OF_BATCHES):
        quizzes, _ = test_generator.__getitem__(batch)
        with Pool() as pool:
            results = pool.map(inference_sudoku, quizzes[0:NUMBER_OF_TESTS])
            for i, pred in enumerate(results):
                idx = batch * NUMBER_OF_TESTS + i
                y_pred2[idx,] = pred

    control = np.zeros((NUMBER_OF_BATCHES * NUMBER_OF_TESTS,
                       y_solutions.shape[1], y_solutions.shape[2]))
    for batch in range(NUMBER_OF_BATCHES):
        start_idx_sols = batch * BATCH_SIZE
        start_idx_control = batch * NUMBER_OF_TESTS
        control[start_idx_control: start_idx_control +
                NUMBER_OF_TESTS] = y_solutions[start_idx_sols: start_idx_sols + NUMBER_OF_TESTS]

    eval_score(y_pred=y_pred2, y_true=control, title=title2)
    end_time = time.perf_counter()
    print(
        f'Prediction of {NUMBER_OF_BATCHES * NUMBER_OF_TESTS} tests completed in {(end_time - start_time):.5f}s')

    # Display Incorrect Sudokus
    # results = y_pred2 == control
    # incorrect_sudokus = []
    # for i,val in enumerate(results):
    # if(not np.all(val)):
    # incorrect_sudokus.append(y_pred2[i,])
    # print(incorrect_sudokus[0])
    # print(control[0])
    return y_pred2, control


if __name__ == '__main__':
    preds, sols = main()
