import sys
sys.path.insert(0, '/home/zachary/dsi/capstone/src/scrapers')
import scrape_tools as st
import pandas as pd
import pickle
from datetime import datetime as dt


def load_dfs():
    fox_df = st.open_as_df('articles_fox')
    hp_df = st.open_as_df('articles_hp')
    reu_df = st.open_as_df('articles_reuters')
    nyt_df = st.open_as_df('articles_nyt')
    ads_df = st.open_as_df('ad_transcripts')


    ads_df=ads_df[ads_df['supports'].isin(('Hillary Clinton','Donald Trump'))]
    ads_df['source'] = ads_df['supports']
    ads_df.drop('supports', axis=1, inplace=True)

    fox_df['date'] = fox_df.date.apply( lambda date_str: dt.date(dt.strptime(date_str, '%Y-%m-%d')))
    fox_df['source'] = 'fox'

    hp_df.drop('author', axis=1, inplace=True)
    hp_df['date'] = hp_df.date.apply( dt.date )
    hp_df['source'] = 'hp'

    reu_df['source'] = 'reu'

    return fox_df, hp_df, reu_df, nyt_df, ads_df


# def load_toy():
#     import pickle
#     with open('../data/toy_data.pkl','rb') as f:
#         data_tup = pickle.load(f)
#     return data_tup


if __name__ == '__main__':
    fox_df, hp_df, reu_df, nyt_df, ads_df = load_dfs()
