import ipdb
from pymongo.errors import DuplicateKeyError, CollectionInvalid
from scrape_tools import open_database_collection, souper, table_grabber, soup_browser, table_to_list
from selenium import webdriver
from datetime import datetime, timedelta
import time
import threading, multiprocessing
from queue import Queue
import numpy as np
from datetime import datetime as dt
import dateutil.parser

def meta_scrape_link(table, link):
    try:
        soup = souper(link, False)
        art_list_soup = soup.find('div', {'class':'page-content-inner clearfix'})
        cards = art_list_soup.find_all('div',{'class':'post-text'})
    except:
        print('page error')
        return

    for card in cards:
        try:
            doc = {}
            h = card.find('h3', {'class':'post-title'})
            a = h.find('a')
            doc['link'] = a['href']
            doc['title'] = a.get_text()
            doc['source'] = 'gp'
            try:
                datestring = card.find('span',{'class':'post-date'}).get_text()
                date = str(dateutil.parser.parse(datestring).date())
            except:
                print('date_error')
                date='None'
            doc['date'] = date
            print(doc['title'])
            table.insert_one(doc)
        except:
            print('card error')
            continue

def meta_scraper(table):

    page_ns = np.arange(1,326)
    links = ['http://thegatewaypundit.com/page/{}/'.format(i) for i in page_ns]

    for link in links:
        thread = threading.Thread(target=meta_scrape_link, args=(table, link))
        thread.start()
        time.sleep(np.random.random()/2+0.5)


def content_adder_thread(table, doc, i):
    link = doc['link']
    title = doc['title']
    _id = doc['_id']

    if 'video' in title.lower():
        table.delete_one(filter={'_id':_id})
        return

    soup = souper(link, False)
    if soup == None:
        print('request error')
        time.sleep(1)
        soup = souper(link, False)
        if soup == None:
            print('request failed twice')
            return
    try:
        entry=soup.find('article',{'class':'clearfix'})
        children = list(entry.children)
        good_children = []
        good_tags = ['p', 'blockquote', 'ul']

        for child in children:
            if child.name in good_tags:
                if 'twitter-tweet' in child.attrs.get('class',[]):
                    continue
                good_children.append(child)

        content = '\n'.join([child.get_text() for child in good_children]).replace('\xa0',' ')


        table.update_one(filter={'_id':_id}, update={'$set':{'content':content}})
        print('{} : {}'.format(i, content[:70]))
    except:
        print('{}: article error'.format(i))


def content_scraper(table):

    docs = table_to_list(table)
    for i,doc in enumerate(docs):
        if 'content' in doc.keys():
            continue
        thread = threading.Thread(name=i, target=content_adder_thread, args=(table, doc, i))
        thread.start()
        time.sleep(np.random.random()/3+0.3)




if __name__ == '__main__':
    table = open_database_collection('gp')
    #meta_scraper(table)
    content_scraper(table)
