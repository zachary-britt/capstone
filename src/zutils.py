import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pprint import pprint
from collections import Counter

from pymongo import MongoClient
import urllib.parse

from pathlib import Path
import os, sys
DATA_PATH = os.environ['DATA_PATH']
import time

from multiprocessing import Pool


'''MONGO UTILS'''

def open_mongo_client():
    username = urllib.parse.quote_plus('admin')
    password = urllib.parse.quote_plus('admin123')
    client = MongoClient('mongodb://%s:%s@127.0.0.1' % (username, password))
    return client


def open_database_collection(collection_name):
    client = open_mongo_client()
    db = client['articles']
    table = db[collection_name]
    return table


def table_grabber(table, field=None):
    '''
    Create a generator over the mongo table which returns the document _id and
    the requested field
    '''
    if field:
        cur = table.find(projection={field:True})
    else:
        cur = table.find()
    while cur.alive:
        yield cur.next()


def table_to_list(table):
    gen = table_grabber(table)
    return list(gen)


def open_as_df(collection_name):
    table = open_database_collection(collection_name)
    t_list = table_to_list(table)
    df = pd.DataFrame(t_list)
    return df



'''DATA LOADING UTILS'''

def make_cat_dict(y, labels):
    y = [{label:value == label for label in labels} for value in y]
    return y

def bi_dict(row, labels):
    d = { label: (row['bias'] if row['orient']==label else 0) for label in labels }
    return d

def make_catbias_dict(df, labels):
    y = np.array(df.apply(bi_dict, axis=1, labels=labels))
    return y


def make_one_hot(y, labels):
    hot_cols = [ np.where(y==label, 1, 0).reshape(-1,1) for label in labels ]
    return hstack(hot_cols)


def tt_split_df(df, test_size, tag_loc):
    new_tags = not tag_loc.exists()
    if new_tags:
        '''perform ttsplit and save object tags for test'''
        inds = df.index.values
        t_inds, e_inds = train_test_split(inds, test_size=test_size)

        tags = df.iloc[e_inds]._id.values
        np.save(tag_loc, tags)
    else:
        # load tags
        tags = np.load(tag_loc)
        # match saved object tags to df _ids
        e_inds = df[np.isin(df._id, tags, assume_unique=True)].index.values
        all_inds = df.index.values
        t_inds = np.setdiff1d(all_inds,e_inds, assume_unique=True)

    # shuffle inds in case classes aren't already shuffled
    np.random.shuffle(t_inds)
    np.random.shuffle(e_inds)

    return df, t_inds, e_inds


def zip_for_spacy(X, y_dict, D=None):
    if D==None:
        return list(zip(X, [{'cats': cats} for cats in y_dict]))
    else:
        return list(zip(X, D, [{'cats': cats} for cats in y_dict]))

def format_y(df, label_type='cats', labels=['left','right']):

    if label_type == 'string':
        y = df.orient.values

    elif label_type == 'cats':
        y = make_cat_dict(df.orient, labels)

    elif label_type == 'one_hot':
        y = make_one_hot(df.orient, labels)

    elif label_type == 'catbias':
        y = make_catbias_dict(df, labels)

    return y


def resampler(y, resampling_type):
    raw_counts = y.value_counts()
    labels = raw_counts.index.values
    counts = Counter({label: raw_counts[label] for label in labels}).most_common()

    label_order, label_counts = zip(*counts)

    new_inds = []

    if resampling_type == 'over':
        N = label_counts[0] # size of largest class
        for label, count in zip(label_order, label_counts):
            label_inds = y[y==label].index.values

            # stuff in at least one of each
            new_inds.extend(label_inds)

            # randomly oversample remainder (N-count == 0 for most represented)
            over_inds = np.random.choice(label_inds, N-count)
            new_inds.extend(over_inds)

    else:   #implicitly undersampling
        N = label_counts[-1]    #size of smallest class
        for label, count in zip(label_order, label_counts):
            label_inds = y[y==label].index.values
            under_inds = np.random.choice(label_inds, N, replace=False)
            new_inds.extend(under_inds)

    new_inds = np.array(new_inds)
    np.random.shuffle(new_inds)
    return new_inds


''' MAIN DATA LOADER '''

def load_and_configure_data(data_name='articles.pkl', **kwargs):
    '''
    Prep formatted dataframe for training or testing

    Performs test train split
        -remembers past training train-test splits to prevent leakage
        -(data points added after tagging get put in training set)
        -default test_size=True

    If prepping test_samples, set test_all=True

    Performs resampling ['over','under','none']

    Configures labels as: ['string','one_hot', or 'dict']
    '''
    label_type = kwargs.get('label_type', 'cats')
    verbose = kwargs.get('verbose',True)
    test_all = kwargs.get('test_all')
    train_all= kwargs.get('train_all')
    labels = kwargs.get('labels',['left','right'])
    get_dates=kwargs.get('dates')
    space_zip = kwargs.get('space_zip', True)
    resampling = kwargs.get('resampling', 'over')


    test_size = kwargs.get('test_size',0.05),
    if test_all:
        return load_and_configure_test_data(data_name, label_type, verbose,
                                            labels, get_dates, space_zip)
    if train_all:
        test_size=0

    data_loc = DATA_PATH + data_name
    df = pd.read_pickle(data_loc)

    if test_size:
        '''Check for tt_split tags'''
        tag_loc = DATA_PATH + 'tag_dir/'+data_name[ :data_name.find('.')]+'_tags.npy'
        tag_loc = Path(tag_loc)
        if not tag_loc.exists():
            if verbose: print('setting train test tags')
        else:
            if verbose: print('using saved train test tags')

        ''' test train split '''
        df, t_inds, e_inds  = tt_split_df(df, test_size, tag_loc)

    else:   # test size is None, all inds are training inds
        t_inds = df.index.values
        e_inds = np.array([])

    y_t = df.iloc[t_inds].orient
    y_e = df.iloc[e_inds].orient

    if verbose:
        print('\ntrain class support:')
        pprint(y_t.value_counts())

        print('\ntest class support:')
        pprint(y_e.value_counts())
        print()


    if resampling == 'over' or resampling == 'under':
        if verbose: print('rebalancing classes via {}sampling'.format(resampling))
        t_inds = resampler(y_t, resampling)
        y_t = y_t[t_inds]


    X_t = df.iloc[t_inds].content.values
    X_e = df.iloc[e_inds].content.values

    # format y values
    y_t = format_y(df.iloc[t_inds], label_type, labels)
    y_e = format_y(df.iloc[e_inds], label_type, labels)

    if verbose:
        print('\nFormatted ys as {}'.format(label_type))
        print('y_t[0] ==', y_t[0], '\n')


    train_data = [X_t, y_t]
    test_data = [X_e, y_e]

    if get_dates:
        '''add dates to return list'''
        D_t = df.iloc[t_inds].date.values
        D_e = df.iloc[e_inds].date.values
        train_data.append(D_t)
        test_data.append(D_e)

    if space_zip:
        ''' zip features to labels '''
        train_data = zip_for_spacy(*train_data)
        test_data = zip_for_spacy(*test_data)

    return {'train':train_data, 'test':test_data}



def peek_tagger(df, tag_loc, peek_size=250, get_peek=False):
    if not tag_loc.exists():
        l_inds = df[df.orient == 'left'].index.values
        r_inds = df[df.orient == 'right'].index.values

        l_peek_inds = np.random.choice(l_inds, peek_size, replace=False)
        r_peek_inds = np.random.choice(r_inds, peek_size, replace=False)

        peek_inds = np.hstack([l_peek_inds, r_peek_inds])

        # save tags
        tags = df.iloc[peek_inds]._id.values
        np.save(tag_loc, tags)

    else:
        # load tags
        tags = np.load(tag_loc)
        # match saved object tags to df _ids
        peek_inds = df[np.isin(df._id, tags, assume_unique=True)].index.values
    all_inds = df.index.values
    test_inds = np.setdiff1d(all_inds, peek_inds, assume_unique=True)

    if get_peek:
        peek_inds.sort()
        return peek_inds
    else:
        np.random.shuffle(test_inds)
        return test_inds


def load_peek_set(data_name='holdout.pkl',label_type='cats', verbose=True,
                    labels=['left','right'], get_dates=False, space_zip=True):
    data_loc = DATA_PATH + data_name
    df = pd.read_pickle(data_loc)

    tag_loc = DATA_PATH + 'tag_dir/'+data_name[ :data_name.find('.')]+'_tags.npy'
    tag_loc = Path(tag_loc)
    if not tag_loc.exists():
        if verbose: print('setting peek tags')
    else:
        if verbose: print('using saved peek tags')

    peek_size = 250
    peek_inds = peek_tagger(df, tag_loc, peek_size=peek_size, get_peek=True )
    df = df.iloc[peek_inds]

    X = df.content.values
    y = format_y(df, label_type='cats')

    if get_dates:
        D = df.date.values
        return_vals.append(D)
    else:
        D = None

    if space_zip:
        return_vals = zip_for_spacy(X,y,D)

    return return_vals


def load_and_configure_test_data(data_name='holdout.pkl', label_type='cats', verbose=True,
                            labels=['left','right'], get_dates=False, space_zip=True):
    data_loc = DATA_PATH + data_name
    df = pd.read_pickle(data_loc)

    tag_loc = DATA_PATH + 'tag_dir/'+data_name[ :data_name.find('.')]+'_tags.npy'
    tag_loc = Path(tag_loc)
    if verbose: print('cutting peeked tags')

    test_inds = peek_tagger(df, tag_loc)
    df = df[test_inds]
    X = df.content.values
    y = format_y(df, label_type, labels)

    return_vals = [X,y]

    if get_dates:
        D = df.date.values
        return_vals.append(D)
    else:
        D = None

    if space_zip:
        return_vals = zip_for_spacy(X,y,D)

    return {'test':return_vals}



''' TERMINAL OUTPUT UTILS'''


def print_progress(title, total, x, start_time, other):
    progress_frac = x/total
    duration = time.time() - start_time+ 0.001
    speed = progress_frac/duration + 0.001
    seconds = int((1 - progress_frac) / speed)
    minutes = int(seconds / 60)
    seconds = seconds % 60
    if minutes:
        time_str = '{:2}:{:2}'.format(minutes,seconds)
    else:
        time_str = '{:2} s'.format(seconds)

    bar_progress = int(progress_frac * 30)
    x_len = len(str(x))
    padding = len(str(total)) - x_len
    sys.stdout.write('\r' +title +": ["+ "=" * (bar_progress) + '>' + '-' * (30-bar_progress) + '] ' +
                                    ' ' * padding + str(x) +' / ' + str(total) +
                                    ', eta: ' + time_str + ', ' + other)
    sys.stdout.flush()



class ProgressBar:
    def __init__(self, title, total):
        self.total = total
        self.title = title
        self.start_time = time.time()
        self.processor = Pool(1)
        print('\nPROGRESS')
        self.progress(0)
        self.n=0

    def progress(self, x, other=''):
        self.processor.apply(print_progress, (self.title, self.total, x, self.start_time, other))

    def increment(self, x=1, other=''):
        self.n += x
        self.progress(self.n, other)

    def kill(self, other=''):
        seconds_dur = int(time.time() - self.start_time)
        minutes = int(seconds_dur / 60)
        seconds = seconds_dur % 60
        if minutes:
            time_str = '{:2}:{:2}'.format(minutes,seconds)
        else:
            time_str = '{:2} s'.format(seconds)
        sys.stdout.write('\r' +self.title +": ["+"=" * 31  + "] "+
                        str(self.total) +' / ' + str(self.total) +'time: '+ time_str +", " +other+ "\n" )
        sys.stdout.flush()
        self.processor.close()
        self.processor.join()
        return seconds_dur

if __name__ == '__main__':

    N = 100
    n = 0
    PB = ProgressBar('Bar', N)
    while n < N:
        n += 1
        PB.progress(n)
        time.sleep(0.02)
    PB.kill()
