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

def _make_cat_dict(y, labels):
    y = [{label:value == label for label in labels} for value in y]
    return y

def _bi_dict(row, labels):
    d = { label: (row['bias'] if row['orient']==label else 0) for label in labels }
    return d

def _make_catbias_dict(df, labels):
    y = np.array(df.apply(_bi_dict, axis=1, labels=labels))
    return y


def _make_one_hot(y, labels):
    hot_cols = [ np.where(y==label, 1, 0).reshape(-1,1) for label in labels ]
    return np.hstack(hot_cols)


def _tt_split_df(df, test_size, tag_loc):
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


def _zipit(X, y_dict, D=None, **cfg):

    if cfg.get('catnest'):
        y_dict = [ {'cats':cats} for cats in y_dict ]

    if D: zargs = [X, D, y_dict]
    else: zargs = [X, y_dict]
    return list(zip(*zargs))


def _format_y(df, **cfg):

    label_type = cfg.get('label_type','cats')
    labels= cfg.get('labels',['left','right'])

    if label_type == 'cats':
        y = _make_cat_dict(df.orient, labels)

    elif label_type == 'one_hot':
        y = _make_one_hot(df.orient, labels)

    elif label_type == 'catbias':
        y = _make_catbias_dict(df, labels)

    else: #label_type == 'string': # (or unrecognized)
        if label_type != 'string':
            print('''label_type unrecognized, formatting as string, options are:\n
                    "cats"\n"one_hot"\n"catbias"\n"string"''')
        y = df.orient.values

    return y


def _resampler(y, **cfg):
    resampling_type=cfg.get('resampling')


    if resampling_type == 'none' or not resampling_type:
        return y.index.values

    valid_types = ['over','under','choose_n']
    if resampling_type not in valid_types:
        print('{} is not a recognized resampling_type')
        return y.index.values

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

    elif resampling_type == 'under':
        N = label_counts[-1]    #size of smallest class
        for label, count in zip(label_order, label_counts):
            label_inds = y[y==label].index.values
            under_inds = np.random.choice(label_inds, N, replace=False)
            new_inds.extend(under_inds)

    elif resampling_type == 'choose_n':
        N = cfg.get('max_class_size')
        if not N:
            print("MUST SPECIFY 'max_class_size' for 'choose_n' resampling")
        for label in label_order:
            label_inds = y[y==label].index.values
            choice_inds = np.random.choice(label_inds, N, replace=True)
            new_inds.extend(choice_inds)

    new_inds = np.array(new_inds)
    np.random.shuffle(new_inds)
    return new_inds


''' MAIN DATA LOADER '''

def load_and_configure_data(data_name, **cfg):
    '''
    Prep formatted dataframe for training or testing

    INPUT:
        data_name   (str)           |   path to dataframe pickle from data dir

    OPTIONAL INPUT:
        label_type  (str)
        verbose     (bool)
        test_size   (float)
        test_all    (bool)
        train_all   (bool)
        labels      (list of str)
        get_dates   (bool)
        zipit       (bool)
        resampling  (str)

    OUTPUT:
        data        (dict)          |   e.g. {  'train':(training_data)
                                                'test': (testing_data)}

    SETUP TEST TRAIN SPLIT:

    The point being to maintain tags on test data so that the model can train,
    you can save the model, and the loader will remember the test set split
    so that the model can resume training without leakage.

    Had to save mongo object _ids as if the cleaning and formatting process
    changes or more data is added, the test set indices would change, while
    the _ids are stable.

    Performs resampling ['over','under','none']


    LABEL FORMATTING:

        df.iloc[0[['orient','bias']] =  orient    right
                                        bias        0.5

        label_type:     |       y[0]:
        'string'        |       'right'
        'one_hot'       |       np.array([0, 1])
        'cats'          |       {'left': False, 'right': True}
        'catbias'       |       {'left': 0, 'right': 0.5}

    '''

    # redirect:
    # if only running evaluation, call test configurer
    if cfg.get('test_all') or cfg.get('peek'):
        return _load_and_configure_test_data(data_name, **cfg)


    # optional argument unpacking
    label_type = cfg.get('label_type', 'cats')
    verbose = cfg.get('verbose',True)
    test_size = cfg.get('test_size',0.05)
    test_all = cfg.get('test_all')
    train_all= cfg.get('train_all')
    labels = cfg.get('labels',['left','right'])
    get_dates=cfg.get('dates')
    zipit = cfg.get('zipit', True)
    resampling = cfg.get('resampling', 'over')


    # if only running training, set test size to 0 to skip test set tagging setup
    if train_all:
        test_size=0


    # load dataframe
    data_loc = DATA_PATH + data_name
    df = pd.read_pickle(data_loc)


    '''SETUP TEST TRAIN SPLIT'''
    if test_size:
        # Check for tt_split tags
        '''
        The point being to maintain tags on test data so that the model can train,
        you can save the model, and the loader will remember the test set split
        so that the model can resume training without leakage.

        Had to save mongo object _ids as if the cleaning and formatting process
        changes or more data is added, the test set indices would change, while
        the _ids are stable.
        '''
        tag_loc = DATA_PATH + 'tag_dir/'+data_name[ :data_name.find('.')]+'_tags.npy'
        tag_loc = Path(tag_loc)
        if not tag_loc.exists():
            if verbose: print('setting train test tags')
        else:
            if verbose: print('using saved train test tags')

        # test train split t for train, e for eval (or validate)
        df, t_inds, e_inds  = _tt_split_df(df, test_size, tag_loc)

    else:   # test size is None, all inds are training inds
        t_inds = df.index.values
        e_inds = np.array([])


    '''SETUP RESAMPLING/CLASS BALANCING'''
    y_t = df.iloc[t_inds].orient
    y_e = df.iloc[e_inds].orient

    if verbose:
        print('\ntrain class support:')
        pprint(y_t.value_counts())

        print('\ntest class support:')
        pprint(y_e.value_counts())
        print()

    if resampling and resampling != 'none':
        if verbose: print('rebalancing classes via {}sampling'.format(resampling))
        t_inds = _resampler(y_t, **cfg)
        y_t = y_t[t_inds]


    X_t = df.iloc[t_inds].content.values
    X_e = df.iloc[e_inds].content.values



    '''LABEL FORMATTING'''
    '''
        df.iloc[0[['orient','bias']] =  orient    right
                                        bias        0.5
        #
        label_type:     |       y[0]:
        'string'        |       'right'
        'one_hot'       |       np.array([0, 1])
        'cats'          |       {'left': False, 'right': True}
        'catbias'       |       {'left': 0, 'right': 0.5}
        ''

    '''
    y_t = _format_y(df.iloc[t_inds], **cfg)
    y_e = _format_y(df.iloc[e_inds], **cfg)

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

    '''ZIPIT'''
    '''
        if zipit data is list of tuple(text[i],{'cats':y[i]})
        if not zipit    data[0] = text list
                        data[1] = y list
    '''
    if zipit:
        ''' zip features to labels '''
        train_data = _zipit(*train_data, **cfg)
        test_data = _zipit(*test_data, **cfg)


    return {'train':train_data, 'test':test_data}



def _peek_tagger(df, tag_loc, peek_size=250, get_peek=False):
    if not tag_loc.exists():
        l_inds = df[df.orient == 'left'].index.values
        r_inds = df[df.orient == 'right'].index.values
        c_inds = df[df.orient == 'center'].index.values

        l_peek_inds = np.random.choice(l_inds, peek_size, replace=False)
        r_peek_inds = np.random.choice(r_inds, peek_size, replace=False)
        c_peek_inds = np.random.choice(c_inds, peek_size, replace=False)

        peek_inds = np.hstack([l_peek_inds, r_peek_inds, c_peek_inds])

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


def _load_peek_set(data_name, **cfg):

    # optional argument unpacking
    label_type = cfg.get('label_type', 'cats')
    verbose = cfg.get('verbose',True)
    labels = cfg.get('labels',['left','right'])
    get_dates=cfg.get('dates')
    zipit = cfg.get('zipit', True)


    data_loc = DATA_PATH + data_name
    df = pd.read_pickle(data_loc)

    tag_loc = DATA_PATH + 'tag_dir/'+data_name[ :data_name.find('.')]+'_tags.npy'
    tag_loc = Path(tag_loc)
    if not tag_loc.exists():
        if verbose: print('setting peek tags')
    else:
        if verbose: print('using saved peek tags')

    peek_size = 250
    peek_inds = _peek_tagger(df, tag_loc, peek_size=peek_size, get_peek=True )
    df = df.iloc[peek_inds]

    X = df.content.values
    y = _format_y(df, **cfg)

    test_data = [X, y]

    if get_dates:
        D = df.date.values
        test_data.append(D)

    if zipit:
        test_data = _zipit(*test_data, **cfg)

    return {'test':test_data}


def _load_and_configure_test_data(data_name, **cfg):
    #redirect
    if cfg.get('peek'):
        return _load_peek_set(data_name, **cfg)


    data_loc = DATA_PATH + data_name
    df = pd.read_pickle(data_loc)

    # optional argument unpacking
    label_type = cfg.get('label_type', 'cats')
    verbose = cfg.get('verbose',True)
    peek = cfg.get('peek')
    labels = cfg.get('labels',['left','right'])
    get_dates=cfg.get('dates')
    zipit = cfg.get('zipit', True)


    tag_loc = DATA_PATH + 'tag_dir/'+data_name[ :data_name.find('.')]+'_tags.npy'
    tag_loc = Path(tag_loc)
    if verbose: print('cutting peeked tags')

    test_inds = _peek_tagger(df, tag_loc)
    df = df.iloc[test_inds]
    X = df.content.values
    y = _format_y(df, **cfg)

    test_data = [X,y]

    if get_dates:
        D = df.date.values
        test_data.append(D)

    if zipit:
        return_vals = _zipit(*test_data, **cfg)

    return {'test':return_vals}


def make_ultra_cross_val(**cfg):

    sources = ['hp','mj','od','ai','reu','nyt','fox','bb','gp','cnn']

    art_df = pd.read_pickle('../data/articles.pkl')
    hold_df = pd.read_pickle('../data/holdout.pkl')
    cnn_df = pd.read_pickle('../data/cnn.pkl')
    all_arts_df = pd.concat([art_df, hold_df])
    reddit_df = pd.read_pickle('../data/reddit.pkl')

    for test_source in sources:
        df_test = all_arts_df[all_arts_df.source == test_source]

        art_df = art_df[art_df.source != test_source]

        '''Set up reddit/ remaining articles mix'''

        reddit_ratio = cfg.get('reddit_ratio', 9)

        art_left = art_df[art_df['orient']=='left']
        art_right = art_df[art_df['orient']=='right']
        art_cent = art_df[art_df['orient']=='center']

        red_left = reddit_df[reddit_df['orient']=='left']
        red_right = reddit_df[reddit_df['orient']=='right']
        red_cent = reddit_df[reddit_df['orient']=='center']

        red_N = min(red_left.shape[0], red_right.shape[0])
        art_N = red_N // reddit_ratio


        art_left = art_left.loc[np.random.choice(art_left.index.values, art_N)]
        red_left = red_left.loc[np.random.choice(red_left.index.values, red_N)]

        art_right = art_right.loc[np.random.choice(art_right.index.values, art_N)]
        red_right = red_right.loc[np.random.choice(red_right.index.values, red_N)]

        art_cent = art_cent.loc[np.random.choice(art_cent.index.values, art_N)]
        red_cent = red_cent.loc[np.random.choice(red_cent.index.values, red_N)]

        df_train = pd.concat([art_left, red_left, art_right, red_right, art_cent, red_cent],
                ignore_index=True)

        inds = df_train.index.values
        np.random.shuffle(inds)

        df_train = df_train.iloc[inds]


''' TERMINAL OUTPUT UTILS'''


def _print_progress_async(title, total, x, start_time, other):
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
    time_str = 'ETA: ' + time_str

    bar_progress = int(progress_frac * 30)
    x_len = len(str(x))
    padding = len(str(total)) - x_len

    bar_str =  "["+ "=" * (bar_progress) + '>' + '-' * (30-bar_progress) + ']'
    progress_count_str = str(x) +' / ' + str(total)

    sys.stdout.write('\r' +title +": "+ bar_str + ' ' * padding + progress_count_str +
                                    ', ' + time_str + ', ' + other)
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
        self.n = x
        self.processor.apply_async(_print_progress_async, (self.title, self.total, x, self.start_time, other))

    def increment(self, x=1, other=''):
        self.n += x
        self.progress(self.n, other)

    def kill(self, other=''):
        self.processor.close()
        self.processor.join()
        x = self.n; total = self.total
        progress_frac = x/total
        bar_progress = int(progress_frac * 30)

        bar_progress = int(progress_frac * 30)
        x_len = len(str(x))
        padding = len(str(total)) - x_len

        bar_str =  "["+ "=" * (bar_progress) + '>' + '-' * (30-bar_progress) + ']'
        progress_count_str = str(x) +' / ' + str(total)

        seconds_dur = int(time.time() - self.start_time)
        minutes = int(seconds_dur / 60)
        seconds = seconds_dur % 60
        if minutes:
            time_str = '{:2}:{:2}'.format(minutes,seconds)
        else:
            time_str = '{:2} s'.format(seconds)

        sys.stdout.write('\r' +self.title +":" + bar_str+" " + progress_count_str
                            +' DUR: '+ time_str +", " +other+ "\n" )
        sys.stdout.flush()
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
