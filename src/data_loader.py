import sys
sys.path.insert(0, '/home/zachary/dsi/capstone/src/scrapers')
import scrape_tools as st
import pandas as pd
import pickle
from datetime import datetime as dt
import ipdb

def load_dfs():
    fox_df = st.open_as_df('fox')
    hp_df = st.open_as_df('hp')
    reu_df = st.open_as_df('reuters')
    # nyt_df = st.open_as_df('nyt')
    # ads_df = st.open_as_df('ads')

    #ipdb.set_trace()

    # ads_df=ads_df[ads_df['supports'].isin(('Hillary Clinton','Donald Trump'))]
    # ads_df['source'] = ads_df['supports']
    # ads_df.drop('supports', axis=1, inplace=True)

    # fox_df['date'] = fox_df.date.apply( lambda date_str: dt.date(dt.strptime(date_str, '%Y-%m-%d')))
    fox_df['bias'] = 'right'

    hp_df.drop('author', axis=1, inplace=True)
    # hp_df['date'] = hp_df.date.apply( dt.date )
    hp_df['date']= hp_df.date.apply( lambda date: dt.strftime(date, '%Y-%m-%d'))
    hp_df['bias'] = 'left'

    reu_df['bias'] = 'center'
    # reu_df['date'] = reu_df.date.apply( dt.date )
    reu_df['date']= reu_df.date.apply( lambda date: dt.strftime(date, '%Y-%m-%d'))

    #return fox_df, hp_df, reu_df, nyt_df, ads_df
    return fox_df, hp_df, reu_df


# def load_toy():
#     import pickle
#     with open('../data/toy_data.pkl','rb') as f:
#         data_tup = pickle.load(f)
#     return data_tup


if __name__ == '__main__':
    fox_df, hp_df, reu_df = load_dfs()
