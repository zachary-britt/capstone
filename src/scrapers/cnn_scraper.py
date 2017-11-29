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

import sys

def insert_card(soup, table):
    try:
        card = soup.find('div',{'class':'cnn-search__result-contents'})
    except:
        print('carding error')
        return
    try:
        h = card.find('h3')
        title = h.get_text().replace('\n','')
        link = h.a['href']
    except:
        print('headline error')
        head=''
    try:
        datestring = card.find('div',{'class','cnn-search__result-publish-date'}).get_text()
        date = str(dateutil.parser.parse(datestring).date())
    except:
        print('date error')
        date = 'none'
    try:
        body = card.find('div',{'class','cnn-search__result-body'})
        contents = body.get_text()
    except:
        print('body error')
        return

    doc = {'title':title,'link':link,'date':date, 'contents':contents}
    doc['source']='cnn'
    table.insert_one(doc)


def super_scrape_link(table, link):

    xpath='/html/body/div[5]/div[2]/div/div[2]/div[2]/div/div[3]'

    try:
        soup = souper(link, x_ind=xpath)
        result_soup = soup.find('div',{'class':'cnn-search__results-list'})
        cards = [card for card in result_soup.children if card != '\n']
    except:
        print('page error')
        return

    for card in cards:
        insert_card(card, table)
    print('Inserted link:{}'.format(link))

def super_scrape(table):
    size=100
    base_link = 'http://www.cnn.com/search?size={}&q=the&category=politics&type=article&'.format(size)
    page_ns = np.arange(2,90)
    froms = (page_ns-1)*size

    links = [base_link+'from={}'.format(f) for f in froms]
    for link in links:
        thread = threading.Thread(target=super_scrape_link, args=(table, link))
        thread.start()
        time.sleep(5+np.random.random()*10)


if __name__ == '__main__':
    table = open_database_collection('cnn')
    super_scrape(table)
    #meta_scraper(table)
    #content_scraper(table)
