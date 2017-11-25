import sys, os
PROJ_PATH = os.environ['PROJ_PATH']
DATA_PATH = os.environ['DATA_PATH']
sys.path.insert(0, PROJ_PATH+'/src/scrapers')
import scrape_tools as st
import pandas as pd
import pickle
from datetime import datetime as dt
import ipdb
import numpy as np

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

    # create bias metric, -1 = left, 0 = center, 1 = right
    od_df['bias'] = -1;     od_df['orient'] = 'left';
    mj_df['bias'] = -1;     mj_df['orient'] = 'left';
    hp_df['bias'] = -0.5;   hp_df['orient'] = 'left'
    reu_df['bias'] = 0;     reu_df['orient']= 'center'
    fox_df['bias'] = 0.5;   fox_df['orient']= 'right'
    bb_df['bias'] = 1;      bb_df['orient'] = 'right'
    ads_df['bias'] = np.where( ads_df.supports=='Donald Trump', 1, -1 )
    ads_df['orient'] = np.where( ads_df.supports=='Donald Trump', 'right', 'left' )


    dfs = {'fox':fox_df, 'hp':hp_df, 'reu':reu_df, 'mj':mj_df, 'ads':ads_df,
            'bb':bb_df, 'od':od_df}

    drops = ['author','source','supports']
    for name in dfs:
        for col in drops:
            try: dfs[name].drop(col, axis=1, inplace=True)
            except: pass
    return dfs


def load_reddit():
    from dateutil import tz

    left_df = st.open_as_df('left_reddit')
    right_df = st.open_as_df('right_reddit')
    neutral_df = st.open_as_df('neutral_reddit')


    left_df['bias'] = -1;   left_df['orient'] = 'left'
    right_df['bias']= 1;    right_df['orient'] = 'right'
    neutral_df['bias']=0;   neutral_df['orient']='center'

    df = pd.concat([left_df,right_df], ignore_axis=True)
    drops = ['author','score','subreddit']
    df.drop(drops, axis=1, inplace=True)
    df.rename(columns={'body':'content','created_utc':'date'})

    return df

# def load_toy():
#     import pickle
#     with open('../data/toy_data.pkl','rb') as f:
#         data_tup = pickle.load(f)
#     return data_tup


if __name__ == '__main__':
    dfs = load_dfs()
