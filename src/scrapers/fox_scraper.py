import ipdb
from pymongo.errors import DuplicateKeyError, CollectionInvalid
from scrape_tools import open_database_collection, souper, table_grabber, soup_browser
from selenium import webdriver
from datetime import datetime
import time

import threading, multiprocessing

def fox_search_link_builder(key_word, min_date, max_date, start=0):
    fox_search_list = ['http://www.foxnews.com/search-results/search?q=',
                        key_word,
                        '&ss=fn&sort=latest&section.path=fnc/politics&type=story&',
                        'min_date=', min_date, '&max_date=', max_date,
                        '&start=', str(start)]
    link = ''.join(fox_search_list)
    return link



def meta_scraper(table, key_word, min_date, max_date):
    #import ipdb; ipdb.set_trace()
    start = 0
    link = fox_search_link_builder(key_word, min_date, max_date, start)

    browser = webdriver.PhantomJS()
    browser.get(link)
    soup = soup_browser(browser)

    hits_str = soup.find('span',{'ng-bind':'numFound'}).text
    hits = int(hits_str.replace(',',''))

    while start < hits:
        link = fox_search_link_builder(key_word, min_date, max_date, start)
        browser.get(link)
        soup = soup_browser(browser)

        items = soup.find_all('div',{'ng-repeat':'article in articles'})

        for i,item in enumerate(items):

            meta = {}
            art_link = item.find('a',{'ng-bind':'article.title'})
            if art_link==None:
                continue
            meta['link'] = art_link['href']
            meta['title'] = art_link.text
            print(start+i,":", meta['title'])

            date_text = item.find('span').text
            date = datetime.strptime(date_text, '%b %d, %Y')
            meta['date'] = str(date.date())

            try:
                table.insert_one(meta)
            except(DuplicateKeyError):
                print('DuplicateKeyError, object already in database:')
                print("-",meta['title'])
            #end for
        start += 10
        time.sleep(1)
        #end while


def get_content(link):
    try:
        soup = souper(link, on_browser=False)
        art_body = soup.find('div', {'class':'article-body'})
        ps = art_body.find_all('p')
        text_lines=[]
        for p in ps:
            text_lines.append(p.get_text())
        content = '\n'.join(text_lines)
        table.update_one(filter={'_id':_id}, update={'$set':{'content':content}})
        print(i, content[:70])
    except:
        continue

def content_adder(table):
    import numpy as np
    gen = table_grabber(table)
    #import ipdb; ipdb.set_trace()
    for i,doc in enumerate(gen):
        if not 'content' in doc.keys():
            link = doc['link']
            _id = doc['_id']

            time.sleep(np.random.random())


if __name__ == '__main__':
    key_word = 'the'
    min_date = '2016-11-01'
    max_date = '2017-11-01'
    table = open_database_collection('articles_fox')
    #meta_scraper(table, key_word, min_date, max_date)
    content_adder(table)
