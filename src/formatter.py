import sys
sys.path.insert(0, '/home/zachary/dsi/capstone/src/scrapers')
import scrape_tools as st
from database_cleaning import table_to_list



def text_formatting():
    fox_table = st.open_database_collection('articles_fox')
    fox = table_to_list(fox_table)

    hp_table = st.open_database_collection('articles_hp')
    hp = table_to_list(hp_table)

    nyt_table = st.open_database_collection('articles_nyt')
    nyt = table_to_list(nyt_table)




if __name__ == '__main__':
    text_formatting()
