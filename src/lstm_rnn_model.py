from keras import backend as K

''' RNN flavors '''

from keras.layers import RNN, GRU, LSTM, StackedRNNCells, CuDNNLSTM, CuDNNGRU

from keras.layers import Masking, Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import losses

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import os
import ipdb

from pprint import pprint


def iter_data_files():
    data_dir = '/home/zachary/dsi/capstone/data/df_w_tensors/'
    for file_ in os.listdir(data_dir):
        yield os.path.join(data_dir, file_)

def load_one():
    data_dir = '/home/zachary/dsi/capstone/data/df_w_tensors/'
    df = pd.read_pickle(data_dir+'df_0.pkl')
    return df

def labeller_bineraizer(y):
    left = np.where(y=='left', 1, 0).reshape(-1,1)
    center = np.where(y=='center', 1, 0).reshape(-1,1)
    right = np.where(y=='right', 1, 0).reshape(-1,1)
    return np.hstack([left, center, right])

def labeller_bineraizer_no_center(y):
    # left = np.where(y=='left', 1, 0).reshape(-1,1)
    # right = np.where(y=='right', 1, 0).reshape(-1,1)

    y = np.where(y=='left', np.array([1,0], np.array([0,1])))
    return y

def de_binarize(y):
    convert = { np.array([1,0,0]):'left',np.array([0,1,0]):'center',np.array([0,0,1]):'right'}
    ys = np.array([ convert[i] for i in y ])
    return ys

def de_binarize(y):
    convert = { np.array([1,0]):'left', np.array([0,1]):'right'}
    ys = np.array([ convert[i] for i in y ])
    return ys

class Net:
    def __init__(self, seed, verbose=True):

        np.random.seed(seed)  # for reproducibility

        self.verbose=verbose


        ''' HYPER PARAMETERS '''
        # number of training samples used at a time to update the weights
        self.batch_size = 64
        # Max number of passes through the entire train dataset (Early stopping
        # prevents unnecessary epochs)
        nb_epoch = 10


        # dense activation function
        dense_activation = 'relu'
        # dense sequence of nodes:
        dense_sequence = [60, 60]

        #learning rate used by atom
        lr = 0.05
        #dropout rate used in dropouts
        self.dr = 0.3

        patience = 3

        X_train, y_train, X_test, y_test = self.load_and_configure_data()

        if self.verbose:
            print('X_train shape:', X_train.shape)
            print(X_train.shape[0], 'train samples')
            print(X_test.shape[0], 'test samples')

        self.model = Sequential()
        # self.build_CuDNN_LSTM()
        self.build_LSTM()
        self.build_layers(dense_sequence, dense_activation)
        self.compile(lr)

        #ipdb.set_trace()

        self.train(X_train, y_train, self.batch_size, nb_epoch, patience)
        self.recorded_score = self.score(X_test, y_test)

    #
    # def check_data_dim(self):
    #
    #     print( "Checking data dim")
    #     df = load_one()
    #     X = df.tensor.values
    #
    #     self.features = X[0].shape[1]
    #     self.timesteps = 1000
    #     self.nb_classes_ = 2


    def load_and_configure_data(self):
        # the data, shuffled and split between train and test sets
        #(X_train, y_train), (X_test, y_test) = mnist.load_data()

        # load just a couple files for now:

        file_load_count = 4

        f_gen = iter_data_files()
        file_paths = []
        for _ in range(file_load_count):
            file_paths.append(next(f_gen))

        from multiprocessing import Pool
        pool = Pool(4)

        print('Loading DataFrames')

        dfs = pool.map( pd.read_pickle, file_paths )
        df = pd.concat(dfs)


        print('DataFrames loaded ')

        df_train, df_test = train_test_split(df, test_size = 0.1)

        ''' padding the sequences to be uniform length '''
        lens = np.array(sorted(df_train.tensor.apply( lambda x: x.shape[0]).values))
        n_samples = lens.shape[0]

        #TODO important hyper param? runtime effect?
        # cutoff_percentile = .80

        #cut off at sequences longer than cutoff_percentile
        # max_len = lens[int(n_samples*cutoff_percentile)]

        max_len = 1000

        X_train = df_train.tensor.values
        X_test = df_test.tensor.values

        #TODO important hyper params?
        X_train = pad_sequences(X_train, max_len , dtype='float32', padding='pre', truncating='pre')
        X_test = pad_sequences(X_test, max_len , dtype='float32', padding='pre', truncating='pre')


        y_train = df_train.bias.values
        y_test = df_test.bias.values


        self.features = X_train[0].shape[1]
        self.timesteps = max_len

        self.nb_classes_ = 3

        y_train = labeller_bineraizer(y_train)
        y_test = labeller_bineraizer(y_test)

        return X_train, y_train, X_test, y_test


    def build_CuDNN_LSTM(self, **kwrgs):



        CuDNN_LSTM_hypers = { 'kernel_initializer':'glorot_uniform',
                                'recurrent_initializer':'orthogonal', 'bias_initializer':'zeros',
                                'unit_forget_bias':True, 'kernel_regularizer':None, 'recurrent_regularizer':None,
                                'bias_regularizer':None, 'activity_regularizer':None, 'kernel_constraint':None,
                                'recurrent_constraint':None, 'bias_constraint':None, 'return_sequences':True,
                                'return_state':False, 'stateful':False, }

        self.model.add(CuDNNLSTM(units=32, input_shape=(self.timesteps, self.features), **CuDNN_LSTM_hypers))

        self.model.add(CuDNNLSTM(units=16, **CuDNN_LSTM_hypers))

        # self.model.add(CuDNNLSTM(units=16, **LSTM_hypers))

    def build_LSTM(self):

        self.model.add(Masking(mask_value=0., input_shape=(self.timesteps, self.features)))

        #
        # LSTM_hypers = {'activation':'tanh', 'recurrent_activation':'hard_sigmoid', 'use_bias':True,
        # 'kernel_initializer':'he_normal', 'recurrent_initializer':'orthogonal', 'bias_initializer':'zeros',
        # 'unit_forget_bias':True, 'kernel_regularizer':None, 'recurrent_regularizer':None, 'bias_regularizer':None,
        # 'activity_regularizer':None, 'kernel_constraint':None, 'recurrent_constraint':None, 'bias_constraint':None,
        # 'dropout':0.1, 'recurrent_dropout':0.1, 'implementation':1, 'return_sequences':False, 'return_state':False,
        # 'go_backwards':False, 'stateful':False, 'unroll':False}

        LSTM_hypers = {'kernel_initializer':'he_normal', 'implementation':2}

        # self.model.add(LSTM(units=32, **LSTM_hypers))
        #
        # self.model.add(LSTM(units=16, **LSTM_hypers))

        #LSTM_hypers['return_sequences'] = False

        self.model.add(LSTM(units=10, **LSTM_hypers ))




    def build_layers(self, sequence, activation):
        # self.model.add(Flatten())
        # if self.verbose: print('Model flattened out to ', self.model.output_shape)

        # for nodes in sequence:
        #     self.model.add(Dense(nodes, activation=activation))
        #     self.model.add(Dropout(self.dr))

        self.model.add(Dense(self.nb_classes_, activation='softmax')) # 3 final nodes



    def compile(self, lr):
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
        pprint(self.model.predict(X_test))
        print('Test accuracy:', score[1]) # this is the one we care about
        return score[1]


if __name__ == "__main__":
    # scores = np.zeros(3)
    # seeds = [1000,2000,3000]
    # for i,seed in enumerate(seeds):
    #     net = Net(seed, verbose=False)
    #     scores[i]=net.recorded_score

    net = Net(seed=200, verbose=True)

    # epochs = 100
    # for i in range epoch:
    #     f_gen = iter_data_files()



    print(net.recorded_score)

    # print( "Averaged Score: ", scores.mean())
