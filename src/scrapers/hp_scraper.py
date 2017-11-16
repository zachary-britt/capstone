import ipdb
from pymongo.errors import DuplicateKeyError, CollectionInvalid
from scrape_tools import open_database_collection, souper, table_grabber, soup_browser
from selenium import webdriver
from datetime import datetime, timedelta
import time
import threading, multiprocessing
from queue import Queue


# def get_hp_archive_pages(min_date, max_date):
#
#     main_link = 'https://www.huffingtonpost.com/archive/'
#
#     start = datetime.strptime(min_date, '%Y-%m-%d')
#     stop = datetime.strptime(max_date, '%Y-%m-%d')
#     date = start
#     dates = []
#     while date <= stop:
#         dates.append(date)
#         date += timedelta(1)
#     date_strings = [ "{}-{}-{}".format(date.year,date.month,date.day) for date in dates]
#
#     links = [main_link + date_str for date_str in date_strings]
#
#     return links, dates

# def get_links_from_page(page_link):
#     soup = souper(page_link)
#     archival_soup = soup.find('div', {'class','archive'}).find('ul')
#     link_soup = archival_soup.find_all('a')
#     links = [i['href'] for i in link_soup]
#     return links

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
                except(DuplicateKeyError):
                    print('DuplicateKeyError')
            except:
                print('oh well')
    except:
        print('bummer that page didnt load!')




def hp_meta_scraper(table):
    import numpy as np
    import ipdb
    #ipdb.set_trace()
    pages = range(95,601)
    base_link = 'https://www.huffingtonpost.com/topic/donald-trump?page='
    page_links = [base_link + str(page) for page in pages]

    gen = table_grabber(table)
    link_set={ doc['link'] for doc in gen}

    for page in page_links:
        time.sleep(np.random.random())
        get_links_from_page(page, table, link_set)

        print(page[-3:])


#
# def mega_scraper(table, min_date, max_date):
#     import ipdb
#     import numpy as np
#
#     pages, dates = get_hp_archive_pages(min_date, max_date)
#     pairs = zip(pages,dates)
#     for pair in pairs:
#         time.sleep(2*np.random.random())
#         page, date = pair[0], pair[1]
#
#         links = get_links_from_page(page)
#         i=0
#
#         for link in links:
#             time.sleep(np.random.random())
#             #ipdb.set_trace()
#             #try:
#             soup = souper(link, False)
#             doc = {}
#
#             if soup == None:
#                 print('empty soup')
#                 continue
#
#             try:
#                 title = soup.find('h1',{'class':'headline__title'}).text
#                 print(title)
#             except:
#                 print('title error')
#                 continue
#
#             try:
#                 section = soup.find('a', {'class':'entry-eyebrow__link bn-department-link bn-clickable'}).text
#                 if section != 'POLITICS\n':
#                     continue
#             except:
#                 print('section error')
#                 continue
#
#             try:
#                 art_body = soup.find('div',{'class':'entry__text js-entry-text bn-entry-text'})
#                 ps = art_body.find_all('p')
#                 text_lines=[]
#                 for p in ps:
#                     text_lines.append(p.get_text())
#                 content = '\n'.join(text_lines)
#             except:
#                 print('art_body error')
#
#             doc['link'] = link
#             doc['content'] = text
#             doc['title'] = title
#             #TODO date is currently weak: search for and parse timestamp__date--published
#             doc['date'] = date
#
#             ipdb.set_trace()
#
#             try:
#                 table.insert_one(doc)
#                 print(i)
#                 i+=1
#
#             except(DuplicateKeyError):
#                 print('DuplicateKeyError, object already in database:')
#                 print("-",doc['title'])
#             # except:
#             #     print('some_error')
#             time.sleep(np.random.random())
#             #end for
#         #end while




def hp_content_collector(table):
    #ipdb.set_trace()
    gen = table_grabber(table)

    for i,doc in enumerate(gen):

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
            print(i, content[:70])
        except:
            print('content error')



    #doc['link'] = link


if __name__ == '__main__':

    # min_date = '2016-11-01'
    # max_date = '2017-11-01'

    table = open_database_collection('articles_hp')
    #hp_meta_scraper(table)
    hp_content_collector(table)



















    #
