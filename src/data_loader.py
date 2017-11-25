import sys
sys.path.insert(0, '/home/zachary/dsi/capstone/src/scrapers')
import scrape_tools as st
import pandas as pd
import pickle
from datetime import datetime as dt
import ipdb
import numpy as np
import os
PROJ_PATH = os.environ['PROJ_PATH']

def fill_dates(df):
    inds_missing = df[df.date=='None' ].index.values
    for i in inds_missing:
        df['date'].iloc[i] = df['date'].iloc[i-1]
    return df



def load_dfs():

    #open mongo collectoins as pandas dataframes
    fox_df = st.open_as_df('fox')
    hp_df = st.open_as_df('hp')
    reu_df = st.open_as_df('reuters')
    mj_df = st.open_as_df('mj')
    bb_df = st.open_as_df('bb')
    od_df = st.open_as_df('od')
    ads_df = st.open_as_df('ads')

    # match up df format
    ads_df=ads_df[ads_df['supports'].isin(('Hillary Clinton','Donald Trump'))]
    hp_df.drop('author', axis=1, inplace=True)
    mj_df = fill_dates(mj_df)
    hp_df['date']= hp_df.date.apply( lambda date: dt.strftime(date, '%Y-%m-%d'))
    reu_df['date']= reu_df.date.apply( lambda date: dt.strftime(date, '%Y-%m-%d'))

    # create bias metric, -1 = left, 0 = neutral, 1 = right
    od_df['bias'] = -1
    mj_df['bias'] = -1
    hp_df['bias'] = -0.5
    reu_df['bias'] = 0
    fox_df['bias'] = 0.5
    bb_df['bias'] = 1
    ads_df['bias'] = np.where( ads_df.supports=='Donald Trump', 1, -1 )

    dfs = {'fox':fox_df, 'hp':hp_df, 'reu':reu_df, 'mj':mj_df, 'ads':ads_df} #'bb':bb_df

    drops = ['author','source','supports']
    for name in dfs:
        for col in drops:
            try: dfs[name].drop(col, axis=1, inplace=True)
            except: pass
    return dfs


# def load_toy():
#     import pickle
#     with open('../data/toy_data.pkl','rb') as f:
#         data_tup = pickle.load(f)
#     return data_tup


if __name__ == '__main__':
    dfs = load_dfs()
