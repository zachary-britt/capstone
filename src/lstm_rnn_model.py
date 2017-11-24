from keras import backend as K

''' RNN flavors '''

from keras.layers import RNN, GRU, LSTM, StackedRNNCells, CuDNNLSTM, CuDNNGRU

from keras.layers import Masking, Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from keras import losses

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

import os
import ipdb

from pprint import pprint


def get_data_files_list(index_range):
    data_dir = '/home/zachary/dsi/capstone/data/df_w_tensors/'
    return [data_dir + 'df_{}.pkl'.format(index) for index in index_range]


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
    left = np.where(y=='left', 1, 0).reshape(-1,1)
    right = np.where(y=='right', 1, 0).reshape(-1,1)
    return np.hstack([left, right])


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


        # dense activation function
        dense_activation = 'relu'
        # dense sequence of nodes:
        dense_sequence = [60, 60]

        #learning rate used by atom
        lr = 0.05
        #dropout rate used in dropouts
        self.dr = 0.3



        self.check_data_dim()

        # if self.verbose:
        #     print('X_train shape:', X_train.shape)
        #     print(X_train.shape[0], 'train samples')
        #     print(X_test.shape[0], 'test samples')

        self.model = Sequential()
        # self.build_CuDNN_LSTM()
        self.build_LSTM()
        self.build_layers(dense_sequence, dense_activation)
        self.compile(lr)


    def check_data_dim(self):

        print( "Checking data dim")
        df = load_one()
        X = df.tensor.values

        self.features = X[0].shape[1]
        self.timesteps = 1000
        self.nb_classes_ = 2


    def load_and_configure_data(self, file_batch, purpose):

        ''' left vs right data loading '''

        print(" Loading data for: {}".format(purpose))

        from multiprocessing import Pool
        pool = Pool(4)

        print('Loading DataFrames from:')
        for path_ in file_batch:
            print("\t",path_)

        dfs = pool.map( pd.read_pickle, file_batch )
        df = pd.concat(dfs)

        pool.close()
        pool.join()

        print('DataFrames loaded ')


        y = df.bias.values
        X = df.tensor.values


        ''' balancing classes, removing center'''

        left_inds = np.argwhere( y == 'left' ).ravel()
        right_inds = np.argwhere( y == 'right' ).ravel()
        center_inds = np.argwhere( y == 'center' ).ravel()

        N_l = left_inds.shape[0]; N_r = right_inds.shape[0]
        N = N_l + N_r

        l_frac = N_l / N; r_frac = N_r / N;
        l_weight = 1 / l_frac; r_weight = 1 / r_frac;

        print('Suport:\n \t left: \t {} \n \t right:\t {}'.format(N_l, N_r))
        print('Weight:\n \t left: \t {} \n \t right:\t {}'.format(l_weight, r_weight))

        lr_inds = np.hstack([left_inds, right_inds])

        np.random.shuffle(lr_inds)

        X, y = X[lr_inds], y[lr_inds]

        sample_weights = np.where(y=='left', l_weight, r_weight)


        '''one hot encoding categories'''
        y = labeller_bineraizer_no_center(y)


        ''' padding the sequences to be uniform length '''
        X = pad_sequences(X, self.timesteps , dtype='float32', padding='pre', truncating='pre')

        return X, y, sample_weights


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
        # adam=Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        # self.model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"] )
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=["accuracy"])


    def get_file_paths(self):


        train_file_inds = np.arange(16)
        training_files = get_data_files_list(train_file_inds)

        training_file_arr = np.array(training_files)
        training_file_batches = list( training_file_arr.reshape(4,4) )

        val_indicies = np.arange(16,18)
        val_files = get_data_files_list(val_indicies)

        test_indicies = np.arange(18,20)
        test_files = get_data_files_list(test_indicies)

        return training_file_batches, val_files, test_files


    def training_data_gen(self, training_file_batches):
        while True:
            for file_batch in training_file_batches:
                X, y, weights =  self.load_and_configure_data(file_batch, 'Training')
                splits = np.arange(0, X.shape[0], self.batch_size)
                batches = splits.shape[0]-1
                print("training batches in file_batch: {}".format(batches))
                for i in range(batches):
                    X_batch = X[splits[i]: splits[i+1]]
                    y_batch = y[splits[i]: splits[i+1]]
                    w_batch = weights[splits[i]: splits[i+1]]
                    yield (X_batch, y_batch, w_batch)


    def val_data_gen(self, val_files):
        while True:
            X, y, weights =  self.load_and_configure_data(val_files, 'Validating')
            splits = np.arange(0, X.shape[0], self.batch_size)
            batches = splits.shape[0]-1
            print("val batches: {}".format(batches))
            for i in range(batches):
                X_batch = X[splits[i]: splits[i+1]]
                y_batch = y[splits[i]: splits[i+1]]
                w_batch = weights[splits[i]: splits[i+1]]
                yield (X_batch, y_batch, w_batch)


    def train(self, nb_epochs):

        training_file_batches, val_files, _ = self.get_file_paths()

        training_gen = self.training_data_gen(training_file_batches)
        val_gen = self.val_data_gen(val_files)

        #sort of a guess, but not super important
        tr_steps_per_epoch =  160
        val_steps = 20

        self.model.fit_generator(training_gen, steps_per_epoch = tr_steps_per_epoch, epochs=nb_epochs,
                  verbose=self.verbose, validation_data=val_gen, validation_steps=val_steps)

    def score(self):

        _, _, test_files = self.get_file_paths(meta_batch_size, max_training_index)

        X, y, _ = self.load_and_configure_data(test_files, 'Testing')

        score = self.model.evaluate(X, y, verbose=self.verbose)
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

    nb_epochs = 10

    net.train(nb_epochs)

    net.score()



    print(net.recorded_score)

    # print( "Averaged Score: ", scores.mean())
