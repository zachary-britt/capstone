import ipdb
from pymongo.errors import DuplicateKeyError, CollectionInvalid
from scrape_tools import open_database_collection, souper, table_grabber, soup_browser
from selenium import webdriver
from datetime import datetime, timedelta
import time
import threading, multiprocessing
from queue import Queue
import numpy as np


def get_links_from_page(page_link, table, link_set):
    soup = souper(page_link, x_ind='//*[@id="zone_twilight_upper"]/div/div[1]/div/div/div[2]/h2/a')

    try:
        body = soup.find('div', {'class':'zone__content bn-zone', 'data-zone':'twilight_upper'})
        cards = soup.find_all('div',{'class':'bn-card card card--autopilot card--media-left card--twilight'})
        for card in cards:
            try:
                section = card.find('h3').text
                if section != 'POLITICS':
                    continue
                headline = card.find('a', {'class':'card__link bn-card-headline bn-clickable'})
                title = headline.get_text()
                print(title)
                link = 'https://www.huffingtonpost.com/'+headline['href']

                if link in link_set:
                    continue

                try:
                    author = card.find('span', {'class':'bn-clickable'}).text
                except:
                    author = None

                doc = {'link': link, 'title':title, 'author':author}
                try:
                    table.insert_one(doc)
                    link_set.add(link)
                    print(title)
                except(DuplicateKeyError):
                    print('DuplicateKeyError')
            except:
                print('oh well')
    except:
        print('bummer that page didnt load!')



def hp_meta_scraper(table):
    from hp_scraper import get_links_from_page
    pages = range(0,19)
    base_link = 'https://www.huffingtonpost.com/topic/congress?page='
    page_links = [base_link + str(page) for page in pages]

    gen = table_grabber(table)
    link_set={ doc['link'] for doc in gen}

    #arg_iter =

    # def mini_func(link):
    #     get_links_from_page(link, table, link_set)



    # pool = multiprocessing.Pool(4)
    # pool.map(mini_func, page_links)
    for page in page_links:
        time.sleep(np.random.random() +0.5)
        # get_links_from_page(page, table, link_set)

        thread = threading.Thread(target=get_links_from_page, args=(page, table, link_set))
        thread.start()

        #print(page)



def hp_content_collector(table, gen):

    while True:
        try:
            doc = next(gen)
        except StopIteration:
            return
        except ValueError:
            time.sleep(0.1)


        if 'content' in doc.keys():
            continue

        link = doc['link']
        _id = doc['_id']
        soup = souper(link, x_ind='//*[@id="us_5a0cb765e4b0c0b2f2f78878"]/div/div[1]/div[4]/div[1]')
        if soup == None:
            continue

        try:
            date_text = soup.find('span',{'class':'timestamp__date--published'}).text[:10]
            date=datetime.strptime(date_text, '%m/%d/%Y')

            art_body = soup.find('div',{'class':'entry__text js-entry-text bn-entry-text'})
            ps = art_body.find_all('p')
            text_lines=[]
            for p in ps:
                text_lines.append(p.get_text())
            content = '\n'.join(text_lines)


            table.update_one(filter={'_id':_id}, update={'$set':{'content':content, 'date':date}})
            print(content[:70])
        except:
            print('content error')

        continue


def concurrent_content_grabber(table, concurrent_threads):

    gen = table_grabber(table)

    for i in range(concurrent_threads):
        thread = threading.Thread(name=i, target=hp_content_collector, args=(table, gen))
        thread.start()
        time.sleep(0.4)



if __name__ == '__main__':

    # max_date = '2017-11-01'

    table = open_database_collection('hp')
    #hp_meta_scraper(table)
    #hp_content_collector(table)
    concurrent_content_grabber(table, concurrent_threads=10)


















    #
