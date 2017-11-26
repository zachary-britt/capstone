import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pprint import pprint
from collections import Counter
from pathlib import Path
import os
DATA_PATH = os.environ['DATA_PATH']


def make_cat_dict(y, labels):
    y = [{label:value == label for label in labels} for value in y]
    return y


def make_one_hot(y, labels):
    hot_cols = [ np.where(y==label, 1, 0).reshape(-1,1) for label in labels ]
    return hstack(hot_cols)



def tt_split_df(df, test_size, tag_loc, new_tags):

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

def format_y(y, label_type, labels):
    if label_type == 'string':
        y = y.values

    if label_type == 'cats':
        y = make_cat_dict(y, labels)

    if label_type == 'one_hot':
        y = make_one_hot(y, labels)

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


def load_and_configure_test_data(data_name='formatted_holdout.pkl', label_type='cats', verbose=True,
                            labels=['left','right'], get_dates=False, space_zip=True):
    data_loc = DATA_PATH + data_name
    df = pd.read_pickle(data_loc)
    y = df.orient.values
    X = df.content.values
    y = format_y(y)

    return_vals = [X,y]

    if get_dates:
        D = df.date.values
        return_vals.append(D)
    else:
        D = None

    if space_zip:
        return_vals = zip_for_spacy(X,y,D)

    return return_vals


def load_and_configure_data(data_name='formatted_arts.pkl', label_type='cats', verbose=True, test_data=False,
                            test_size=0.2, labels=['left','right'],
                            get_dates=False, space_zip=True, resampling='over'):
    '''
    Prep formatted dataframe for training or testing

    Performs test train split
        -remembers past training train-test splits to prevent leakage
        -(data points added after tagging get put in training set)
        -default test_size=True

    If prepping test_samples, set test_data=True

    Performs resampling ['over','']

    Configures labels as: ['string','one_hot', or 'dict']
    '''

    if test_data:
        return load_and_configure_test_data_(data_name, label_type, verbose,
                                            labels, get_dates, space_zip)

    data_loc = DATA_PATH + data_name

    df = pd.read_pickle(data_loc)

    '''Check for tt_split tags'''
    tag_loc = DATA_PATH + 'tag_dir/'+data_name[ :data_name.find('.')]+'_tags.npy'
    tag_loc = Path(tag_loc)
    if not tag_loc.exists():
        if verbose: print('setting train test tags')
        new_tags=True
    else:
        if verbose: print('using saved train test tags')
        new_tags=False



    ''' test train split '''
    df, t_inds, e_inds  = tt_split_df(df, test_size, tag_loc, new_tags)

    y_t = df.iloc[t_inds].orient
    y_e = df.iloc[e_inds].orient

    if verbose:
        print('\ntrain class support:')
        pprint(y_t.value_counts())

        print('\ntest class support:')
        pprint(y_e.value_counts())
        print()


    if resampling:
        if verbose: print('rebalancing classes via {}sampling'.format(resampling))
        t_inds = resampler(y_t, resampling)
        y_t = y_t[t_inds]

        if verbose:
            print('After resampling: ')
            print('train class support:')
            pprint(y_t.value_counts())

    X_t = df.iloc[t_inds].content.values
    X_e = df.iloc[e_inds].content.values

    # format y values
    y_t = format_y(y_t, label_type, labels)
    y_e = format_y(y_e, label_type, labels)

    if verbose:
        print('\nFormatted ys as {}'.format(label_type))
        print('y_t[0] ==', y_t[0], '\n')

    return_vals = [X_t, X_e, y_t, y_e]

    if get_dates:
        '''add dates to return list'''
        D_t = df.iloc[t_inds].date.values
        D_e = df.iloc[e_inds].date.values
        return_vals.extend([D_t, D_e])

    if space_zip:
        '''ignores dates!'''
        train_data = zip_for_spacy(X_t, y_t)
        test_data = zip_for_spacy(X_e, y_e)
        return_vals = [train_data, test_data]

    return return_vals
