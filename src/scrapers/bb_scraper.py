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
        art_list_soup = soup.find('div', {'class':'article-list'})
        soup_heads = art_list_soup.find_all('article')
    except:
        print('page error')
        return

    for soup_head in soup_heads:
        try:
            doc = {}
            hl = soup_head.find('h2', {'class': 'title'})
            a = hl.find('a')
            doc['link'] = a['href']
            doc['title'] = a.get_text().replace('\n','').replace('\t','').strip(' ')
            doc['source'] = 'bb'
            try:
                date_soup = soup_head.find('span', {'class':'bydate'})
                datestring = date_soup.get_text()
                date = str(dateutil.parser.parse(datestring).date())
            except:
                print('date_error')
                date='None'
            doc['date'] = date
            table.insert_one(doc)


        except:
            print('card error')
            continue

def meta_scraper(table):

    links = ['http://www.breitbart.com/big-journalism/page/{}/'.format(i) for i in np.arange(96)]
    links.extend( ['http://www.breitbart.com/big-government/page/{}/'.format(i) for i in np.arange(633)] )

    for link in links:
        thread = threading.Thread(target=meta_scrape_link, args=(table, link))
        thread.start()
        time.sleep(np.random.random()/3+0.1)
        print('scraped {}'.format(link))


def content_adder_thread(table, doc, i):
    link = doc['link']

    soup = souper(link, False)
    if soup == None:
        print('request error')
        time.sleep(1)
        soup = souper(link, False)
        if soup == None:
            print('request failed twice')
            return
    try:
        body = soup.find( 'div', {'id':"MainW"})

        #concat nested entry-contents
        entry = body.find('div',{'class':'entry-content'})

        # the first time I've ever actually found recursion useful
        def expand_nest(entry):
            children = list(entry.children)
            good_children = []
            good_tags = ['p', 'blockquote', 'ul']

            for child in children:
                if child.name in good_tags:
                    if ['twitter-tweet'] in child.attrs.values():
                        continue
                    good_children.append(child)
                if child.name == 'div' and child.attrs == {'class':'entry-content'}:
                    good_children.append( expand_nest(child))
            return good_children

        children = expand_nest(entry)

        content = '\n'.join([child.get_text() for child in children])
        content = content.replace('\xa0', ' ').replace("'", "’").replace('\u200a',' ')

        _id = doc['_id']
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
        time.sleep(np.random.random()+0.5)


if __name__ == '__main__':
    table = open_database_collection('bb')
    #meta_scraper(table)
    content_scraper(table)
