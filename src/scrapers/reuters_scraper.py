from scrape_tools import open_database_collection, souper, table_grabber, soup_browser, table_to_list
import threading, multiprocessing
import requests
from bs4 import BeautifulSoup as bs
import ipdb
from datetime import datetime as dt
import time
import numpy as np


def build_reuters_search_links():
    links=[]
    for page in range(35,711):
        links.append('https://www.reuters.com/news/archive/politicsNews?view=page&page={}&pageSize=10'.format(page))
    return links

def meta_scraper(table, links):
    for i,link in enumerate(links):
        print("\n\npage: {}\n\n".format(i))
        thread = threading.Thread(name=i, target=meta_scraper_thread, args=(table, link))
        thread.start()
        time.sleep(np.random.random()/3)
        #meta_scraper_thread(table, link)


def meta_scraper_thread(table, link):
    # ipdb.set_trace()

    soup = souper(link, False)
    if soup == None:
        print('request error')

        time.sleep(1)
        soup = souper(link, False)
        if soup == None:
            print('request failed twice')
            return

    try:
        body_soup = soup.find('div', {'class':'news-headline-list'})
        arts = body_soup.find_all('article')
    except:
        print('carding error')
        return

    for art in arts:
        try:
            card = art.find('div',{'class':'story-content'})
            a = card.find('a')
            doc = {}

            doc_link = 'https://www.reuters.com'+a['href']
            doc['link'] = doc_link

            title = a.get_text().replace('\t','').replace('\n','')
            doc['title'] = title

            date_str = card.find('span',{'class':'timestamp'}).get_text()
            date = dt.strptime(date_str, '%b %d %Y')
            #date = dt.date(date)
            doc['date'] = date

            table.insert_one(doc)
            print(dt.date(date), title)
        except:
            print('article error')


def content_adder(table):
    #ipdb.set_trace()

    docs = table_to_list(table)

    for i,doc in enumerate(docs):
        if 'content' in doc.keys():
            continue
        thread = threading.Thread(name=i, target=content_adder_thread, args=(table, doc, i))
        thread.start()
        time.sleep(np.random.random()/3+0.1)
        #content_adder_thread(table, doc)


def content_adder_thread(table, doc, i):
    #ipdb.set_trace()

    link = doc['link']
    if 'content' in doc.keys():
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
        art_body= soup.find( 'div', {'class':"TwoColumnLayout_container_385r0 Article_content_3kpRX TwoColumnLayout_fluid-left_3DYLH"})
        ps = art_body.find_all('p')
        content = '\n'.join([p.get_text() for p in ps])
        _id = doc['_id']
        table.update_one(filter={'_id':_id}, update={'$set':{'content':content}})
        print('{} : {}'.format(i, content[:70]))
    except:
        print('{}: article error'.format(i))

if __name__ == '__main__':

    table = open_database_collection('articles_reuters')
    #links = build_reuters_search_links()
    #meta_scraper(table, links)
    content_adder(table)
