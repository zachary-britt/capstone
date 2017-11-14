import requests, os, ipdb
from pymongo import MongoClient
# from pymongo.errors import DuplicateKeyError, CollectionInvalid
import urllib.parse
from bs4 import BeautifulSoup
import time


def open_mongo_client():
    username = urllib.parse.quote_plus('admin')
    password = urllib.parse.quote_plus('admin123')
    client = MongoClient('mongodb://%s:%s@127.0.0.1' % (username, password))
    return client

def open_database_collection(name):
    client = open_mongo_client()
    db = client['news']
    table = db[name]
    return table

def souper(link):
    try:
        html = requests.get(link).text
        soup = BeautifulSoup(html, 'html.parser')
        return soup
    except:
        return None

def saved_souper(filename):
    import codecs
    f = codecs.open(filename, 'r')
    html = f.read()
    soup = BeautifulSoup(html, 'html.parser')
    return soup

def meta_scraper(table):
    # link = 'https://newrepublic.com/political-ad-database'
    soup = saved_souper('../../data/complete.html')
    ad_items = soup.find_all('div',{'class' : "campaign-ads-ad"})

    for item in ad_items:
        meta = {}
        meta['link'] = item.find('a', {'class':"campaign-ad-link"} )['href']
        meta['title'] = item.find('h3').text
        meta['supports'] = item.find('h4').text
        meta['date'] = item.find('time')['datetime']
        table.insert_one(meta)

def link_generator(table):
    cur = table.find(projection={'link':True})
    while cur.alive:
        yield cur.next()

def transcript_adder(table):
    ipdb.set_trace()
    gen = link_generator(table)
    for item in gen:
        link = item['link']
        soup = souper(link)
        print(soup)

if __name__ == '__main__':
    table = open_database_collection('ad_transcripts')
    #meta_scraper(table)
    transcript_adder(table)
