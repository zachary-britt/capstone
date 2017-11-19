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

def table_to_list(table):
    gen = st.table_grabber(table)
    return list(gen)


if __name__ == '__main__':
    table = st.open_database_collection('articles_reuters')
    remove_dups(table)
