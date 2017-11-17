import sys
sys.path.insert(0, '/home/zachary/dsi/capstone/src/scrapers')
import scrape_tools as st
from database_cleaning import table_to_list
import pandas as pd
import formatter

def loader():

    fox_table = st.open_database_collection('articles_fox')
    fox = table_to_list(fox_table)
    df_fox = pd.DataFrame(df_fox)
    #mini_fox = fox[:100]

    hp_table = st.open_database_collection('articles_hp')
    hp = table_to_list(hp_table)
    mini_hp = hp[:100]

    nyt_table = st.open_database_collection('articles_nyt')
    nyt = table_to_list(nyt_table)
    mini_nyt = nyt[:100]

    ad_table = st.open_database_collection('ad_transcripts')
    ads = table_to_list(ad_table)
    mini_ads = ads[:100]

    return fox, hp, nyt, ads

    # data_list = [mini_ads, mini_fox, mini_hp, mini_nyt]
    # with open('toy_data.pkl', 'wb') as f:
    #     pickle.dump(data_list, f)

def loader_formatter():

    fox, hp, nyt, ads = loader()
