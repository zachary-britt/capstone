import ipdb
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, CollectionInvalid
import datetime as dt
import urllib.parse
from bs4 import BeautifulSoup
import time

'''
I forgot to include the date field in the initial NYT scrape
However the URL has the date within it, so this iterates through the
mongo documents and adds a "date" field using the url

'''

def open_mongo_client():
    username = urllib.parse.quote_plus('admin')
    password = urllib.parse.quote_plus('admin123')
    client = MongoClient('mongodb://%s:%s@127.0.0.1' % (username, password))
    return client

def open_database_collection():
    client = open_mongo_client()
    db = client['news']
    table = db['articles']
    return table

def doc_generator(table):
    cur = table.find(projection={'web_url':True})
    while cur.alive:
        yield cur.next()

def get_date(document):
    url = document['web_url']
    return url[24:34]

def get_id(document):
    _id = document['_id']
    return _id

def iter_through_table(table):
    gen = doc_generator(table)
    for doc in gen:
        date=get_date(doc)
        _id =get_id(doc)
        table.update_one(filter={'_id':_id}, update={'$set':{'date':date}})



if __name__ == '__main__':
    table = open_database_collection()
    iter_through_table(table)
