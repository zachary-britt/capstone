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

def meta_scrape_link(table, link):
    try:
        soup = souper(link, False)
        art_list_soup = soup.find('ul', {'class':'articles-list'})
        soup_heads = art_list_soup.find_all('h3', {'class':'hed'})
    except:
        print('page error')
        return

    for soup_head in soup_heads:
        try:
            doc = {}
            a = soup_head.find('a')
            doc['link'] = a['href']
            doc['title'] = a.get_text().replace('\n','').replace('\t','').strip(' ')
            doc['source'] = 'mj'

            table.insert_one(doc)
        except:
            print('card error')
            continue

def meta_scraper(table):

    page_ns = np.arange(110)
    links = ['http://www.motherjones.com/politics/page/{}/'.format(i) for i in page_ns]

    for link in links:
        meta_scrape_link(table, link)
        time.sleep(np.random.random()/3 +0.1)
        print('scraped {}'.format(link))


def content_adder_thread(table, doc, i):
    link = doc['link']
    # if 'date' in doc.keys():
    #     return

    soup = souper(link, False)
    if soup == None:
        print('request error')
        time.sleep(1)
        soup = souper(link, False)
        if soup == None:
            print('request failed twice')
            return
    try:
        art_body= soup.find( 'article', {'class':"entry-content"})
        ps = art_body.find_all('p')
        ps = [ p for p in ps if p.attrs == {}]
        content = '\n'.join([p.get_text() for p in ps]).replace('\xa0',' ')

        try:
            date_soup = soup.find('span', {'class':'dateline'})
            date_text = date_soup.get_text()
            date=str(dt.strptime(date_text[:13], '%b. %d, %Y').date())
        except:
            try:
                date_soup = soup.find('span', {'class':'dateline'})
                date_text = date_soup.get_text()
                date=str(dt.strptime(date_text[:12], '%b. %d, %Y').date())
            except:
                print('date_error')
                date='None'

        _id = doc['_id']
        table.update_one(filter={'_id':_id}, update={'$set':{'content':content, 'date':date}})
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
        time.sleep(np.random.random()/3+0.1)




if __name__ == '__main__':
    table = open_database_collection('mj')
    #meta_scraper(table)
    content_scraper(table)
