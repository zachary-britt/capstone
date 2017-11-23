from keras import backend as K
from keras.layers import CuDNNLSTM, Masking, Dense, Dropout, Activation
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import os
import ipdb

def iter_data_files():
    data_dir = '/home/zachary/dsi/capstone/data/df_w_tensors/'
    for file_ in os.listdir(data_dir):
        yield os.path.join(data_dir, file_)


class Net:
    def __init__(self, seed, verbose=True):

        np.random.seed(seed)  # for reproducibility

        self.verbose=verbose


        ''' HYPER PARAMETERS '''
        # number of training samples used at a time to update the weights
        self.batch_size = 30
        # Max number of passes through the entire train dataset (Early stopping
        # prevents unnecessary epochs)
        nb_epoch = 10


        # dense activation function
        dense_activation = 'relu'
        # dense sequence of nodes:
        dense_sequence = [30, 30]

        #learning rate used by atom
        lr = 0.005
        #dropout rate used in dropouts
        self.dr = 0.3


        X_train, y_train, X_test, y_test = self.load_and_configure_data()

        if self.verbose:
            print('X_train shape:', X_train.shape)
            print(X_train.shape[0], 'train samples')
            print(X_test.shape[0], 'test samples')

        self.model = Sequential()
        self.build_lstm()
        self.build_layers(dense_sequence, dense_activation)
        self.compile(lr)

        #ipdb.set_trace()

        self.train(X_train, y_train, self.batch_size, nb_epoch, patience = 3)
        self.recorded_score = self.score(X_test, y_test)



    def load_and_configure_data(self):
        # the data, shuffled and split between train and test sets
        #(X_train, y_train), (X_test, y_test) = mnist.load_data()

        # load just a couple files for now:

        f_gen = iter_data_files()
        f1 = next(f_gen)
        f2 = next(f_gen)

        df_1 = pd.read_pickle(f1)
        df_2 = pd.read_pickle(f2)

        df = pd.concat([df_1,df_2])

        df_train, df_test = train_test_split(df, test_size = 0.2)

        ''' Masking and padding the sequences to be uniform length '''
        lens = sorted(X_train.tensor.apply( lambda x: x.shape[0]).values)
        n_samples = lens.shape[0]

        cutoff_percentile = 0.95

        #cut off at sequences longer than cutoff_percentile
        max_len = lens[int(n_samples*cutoff_percentile)]






        X_train = df_train.tensor.values
        X_test = df_test.tensor.values




        y_train = df_train.bias.values
        y_test = df_test.bias.values

        ipdb.set_trace()

        self.vec_dim = X_train[0].shape[1]

        self.nb_classes_ = 3

        def labeller_bineraizer(y):
            left = np.where(y=='left', 1, 0).reshape(-1,1)
            center = np.where(y=='center', 1, 0).reshape(-1,1)
            right = np.where(y=='right', 1, 0).reshape(-1,1)
            return np.hstack([left, center, right])

        y_train = labeller_bineraizer(y_train)
        y_test = labeller_bineraizer(y_test)

        # # convert class vectors to binary class matrices
        # y_train = np_utils.to_categorical(y_train, self.nb_classes_)
        # y_test = np_utils.to_categorical(y_test, self.nb_classes_)

        return X_train, y_train, X_test, y_test

    def build_lstm(self, **kwrgs):
        self.model.add(CuDNNLSTM(10,
            return_sequences=False,
            stateful=False,
            kernel_initializer='he_normal',
            input_shape=(None, self.vec_dim)))


    def build_layers(self, sequence, activation):
        # self.model.add(Flatten()) # necessary to flatten before going into conventional dense layer
        # if self.verbose: print('Model flattened out to ', self.model.output_shape)

        # now start a typical, fully connected neural network
        for nodes in sequence:
            self.model.add(Dense(nodes, activation=activation))
            self.model.add(Dropout(self.dr))

        self.model.add(Dense(self.nb_classes_, activation='softmax')) # 3 final nodes



    def compile(self, lr=0.01):
        # many optimizers available
        # see https://keras.io/optimizers/#usage-of-optimizers
        # suggest you keep loss at 'categorical_crossentropy' for this multiclass problem,
        # and metrics at 'accuracy'
        # suggest limiting optimizers to one of these: 'adam', 'adadelta', 'sgd'
        # how are we going to solve and evaluate it:

        adam=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"] )

    def train(self, X_train, y_train, batch_size, nb_epoch, patience):

        # stops training when validation accuracy hasn't improved in "patience" epochs
        early_stopping=EarlyStopping(monitor='val_acc', min_delta=0,
            patience=patience, verbose=0, mode='auto')

        # during fit process watch train and test error simultaneously
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
                  verbose=self.verbose, validation_split=0.2, callbacks = [early_stopping])

    def score(self, X_test, y_test):
        score = self.model.evaluate(X_test, y_test, verbose=self.verbose)
        # misses = np.sum self.model.predict
        # print('Test loss:', score[0])
        print('Test accuracy:', score[1]) # this is the one we care about
        return score[1]


if __name__ == "__main__":
    # scores = np.zeros(3)
    # seeds = [1000,2000,3000]
    # for i,seed in enumerate(seeds):
    #     net = Net(seed, verbose=False)
    #     scores[i]=net.recorded_score

    net = Net(seed=100, verbose=True)
    print(net.recorded_score)

    # print( "Averaged Score: ", scores.mean())
