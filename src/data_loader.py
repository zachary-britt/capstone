import sys
sys.path.insert(0, '/home/zachary/dsi/capstone/src/scrapers')
import scrape_tools as st
from database_cleaning import table_to_list
import pandas as pd
# import formatter
import pickle

def load():
    import sys
    sys.path.insert(0, '/home/zack/dsi/capstone/src/scrapers')
    import scrape_tools as st
    from database_cleaning import table_to_list
    import pandas as pd
    import pickle

    fox_table = st.open_database_collection('articles_fox')
    fox = table_to_list(fox_table)
    # df_fox = pd.DataFrame(df_fox)
    mini_fox = fox[:100]

    hp_table = st.open_database_collection('articles_hp')
    hp = table_to_list(hp_table)
    mini_hp = hp[:100]

    nyt_table = st.open_database_collection('articles_nyt')
    nyt = table_to_list(nyt_table)
    mini_nyt = nyt[:100]

    ad_table = st.open_database_collection('ad_transcripts')
    ads = table_to_list(ad_table)
    mini_ads = ads[:100]

    data_list = [mini_ads, mini_fox, mini_hp, mini_nyt]
    with open('../data/toy_data.pkl', 'wb') as f:
        pickle.dump(data_list, f)

    return fox, hp, nyt, ads



def load_toy():
    import pickle
    with open('../data/toy_data.pkl','rb') as f:
        data_tup = pickle.load(f)
    return data_tup

def loader_formatter():

    fox, hp, nyt, ads = loader()

if __name__ == '__main__':
    loader()
