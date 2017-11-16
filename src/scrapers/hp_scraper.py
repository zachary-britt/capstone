import ipdb
from pymongo.errors import DuplicateKeyError, CollectionInvalid
from scrape_tools import open_database_collection, souper, table_grabber, soup_browser
from selenium import webdriver
from datetime import datetime, timedelta
import time

def get_hp_archive_pages(min_date, max_date):

    main_link = 'https://www.huffingtonpost.com/archive/'

    start = datetime.strptime(min_date, '%Y-%m-%d')
    stop = datetime.strptime(max_date, '%Y-%m-%d')
    date = start
    dates = []
    while date <= stop:
        dates.append(date)
        date += timedelta(1)
    date_strings = [ "{}-{}-{}".format(date.year,date.month,date.day) for date in dates]

    links = [main_link + date_str for date_str in date_strings]

    return links, dates

def get_links_from_page(page_link):
    soup = souper(page_link)
    archival_soup = soup.find('div', {'class','archive'}).find('ul')
    link_soup = archival_soup.find_all('a')
    links = [i['href'] for i in link_soup]
    return links

def mega_scraper(table, min_date, max_date):
    import ipdb
    import numpy as np

    pages, dates = get_hp_archive_pages(min_date, max_date)
    pairs = zip(pages,dates)
    for pair in pairs:
        time.sleep(2*np.random.random())
        page, date = pair[0], pair[1]

        links = get_links_from_page(page)
        i=0

        for link in links:
            time.sleep(np.random.random())
            ipdb.set_trace()
            #try:
            soup = souper(link, False)
            doc = {}

            if soup == None:
                print('empty soup')
                continue

            try:
                title = soup.find('h1',{'class':'headline__title'}).text
                print(title)
            except:
                print('title error')
                continue

            try:
                section = soup.find('a', {'class':'entry-eyebrow__link bn-department-link bn-clickable'}).text
                if section != 'POLITICS\n':
                    continue
            except:
                print('section error')
                continue

            try:
                art_body = soup.find('div',{'class':'entry__text js-entry-text bn-entry-text'})
                ps = art_body.find_all('p')
                text_lines=[]
                for p in ps:
                    text_lines.append(p.get_text())
                content = '\n'.join(text_lines)
            except:
                print('art_body error')

            doc['link'] = link
            doc['content'] = text
            doc['title'] = title
            #TODO date is currently weak: search for and parse timestamp__date--published
            doc['date'] = date

            ipdb.set_trace()

            try:
                table.insert_one(doc)
                print(i)
                i+=1

            except(DuplicateKeyError):
                print('DuplicateKeyError, object already in database:')
                print("-",doc['title'])
            # except:
            #     print('some_error')
            time.sleep(np.random.random())
            #end for
        #end while


if __name__ == '__main__':

    min_date = '2016-11-01'
    max_date = '2017-11-01'

    table = open_database_collection('articles_hp')

    mega_scraper(table, min_date, max_date)


















    #
