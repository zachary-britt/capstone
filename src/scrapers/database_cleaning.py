import scrape_tools as st
import ipdb
from collections import Counter
from pymongo import MongoClient
import pickle

def remove_dups(table):
    ipdb.set_trace()
    gen = st.table_grabber(table)
    docs = list(gen)
    # urls = [ doc['link'] for doc in docs]
    # _ids = [ doc['_id'] for doc in docs]

    pairs = [ (doc['link'],doc['_id']) for doc in docs]
    pair_dict = dict(pairs)
    id_keepers = set(pair_dict.values())
    id_all = {doc['_id'] for doc in docs}

    kill_ids = id_all.difference(id_keepers)

    for _id in kill_ids:
        table.delete_one(filter={'_id':_id})


def remove_contentless(table):

    #db.articles_fox.deleteMany({content: {$exists: false}})
    pass
    # ipdb.set_trace()
    # gen = st.table_grabber(table)
    # docs = list(gen)
    # # urls = [ doc['link'] for doc in docs]
    # # _ids = [ doc['_id'] for doc in docs]
    #
    # pairs = [ (doc['content'],doc['_id']) for doc in docs]
    #
    #
    #
    # kill_ids = [ pair[1] ]
    #
    # for _id in kill_ids:
    #     table.delete_one(filter={'_id':_id})

def table_to_list(table):
    gen = st.table_grabber(table)
    return list(gen)


if __name__ == '__main__':
    fox_table = st.open_database_collection('articles_fox')
    fox = table_to_list(fox_table)
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
    with open('toy_data.pkl', 'wb') as f:
        pickle.dump(data_list, f)



    #remove_dups(nyt_table)




    # fox_clean()
