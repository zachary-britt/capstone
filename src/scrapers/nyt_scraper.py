import requests, json, os, ipdb, sys
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, CollectionInvalid
import datetime as dt
import urllib.parse
from bs4 import BeautifulSoup
import time
from scrape_tools import open_database_collection, souper, table_grabber
import numpy as np

def nyt_parser(link):
    html = requests.get(link).content
    soup = BeautifulSoup(html, 'html.parser')
    article_content = '\n'.join([i.text for i in soup.select('p.story-body-text')])
    return article_content

class NYTScraper:
    def __init__(self):
        self.i=0
        self.table = open_database_collection('articles_nyt')

        gen = table_grabber(self.table,'web_url')
        self.seen_urls = { doc['web_url'] for doc in gen }



        NYT_API_KEY = os.environ['NYT_API_KEY']
        self.link = 'http://api.nytimes.com/svc/search/v2/articlesearch.json'
        self.payload = {'api-key': NYT_API_KEY}
        self._set_filters()

    def _set_filters(self):
        self.payload['fl']='web_url, source, headline, document_type, news_desk, _id, byline'
        self.payload['fq']='source:(The New York Times) AND news_desk:(Foreign, Editorial, Environment, Opinion, Politics, OpEd, U.S., Washington, World)'

    def _set_date(self,date):
        self.date=date
        self.payload['end_date'] = str(date).replace('-','')
        self.payload['begin_date'] = str(date - dt.timedelta(days=1)).replace('-','')

    def _page_loop(self, total_pages):
        for page in range(total_pages):
            self.payload['page'] = str(page)
            response = self._query()

            if response == None:
                time.sleep(1)
                response = self._query()
                if response == None:
                    continue

            meta_data_list = response['response']['docs']


            for doc in meta_data_list:
                try:
                    link = doc['web_url']
                    self.i += 1
                    if link in self.seen_urls:
                        continue
                    else:
                        self.seen_urls.add(link)

                    html = requests.get(link).content
                    soup = BeautifulSoup(html, 'html.parser')
                    article_content = '\n'.join([i.text for i in soup.select('p.story-body-text')])
                    if not article_content:
                        continue
                    doc['content'] = article_content
                    doc['date'] = self.date
                    self.table.insert_one(doc)

                    print(self.i, ':', article_content[:80])
                    time.sleep(np.random.random())
                except DuplicateKeyError:
                    print ('Duplicate Keys')
            time.sleep(1)


    def _query(self):
        response = requests.get(self.link, params=self.payload)
        if response.status_code != 200:
            print ('WARNING', response.status_code)
            return None
        else:
            return response.json()


    def collect_data(self, min_date, max_date):

        date = min_date
        dates = []
        while date <= max_date:
            dates.append(date)
            date += dt.timedelta(1)
            print("Date: {}".format(date))
            self._set_date(date)
            content = self._query()
            hits = content['response']['meta']['hits']
            print("hits: {}".format(hits))
            total_pages = int(hits / 10) + 1
            self._page_loop(total_pages)


if __name__ == '__main__':
    scraper=NYTScraper()
    max_date = dt.datetime(2017, 11, 1)
    min_date = dt.datetime(2016, 11, 1)
    scraper.collect_data(min_date, max_date)
