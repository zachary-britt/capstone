import requests, json, os, ipdb, sys
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, CollectionInvalid
import datetime as dt
import urllib.parse
from bs4 import BeautifulSoup
import time

def open_mongo_client():
    username = urllib.parse.quote_plus('admin')
    password = urllib.parse.quote_plus('admin123')
    client = MongoClient('mongodb://%s:%s@127.0.0.1' % (username, password))
    return client

def open_database_collection():
    client = open_mongo_client()
    db = client['news']
    table = db['articles']
    return table

class NYTScraper:

    def __init__(self, table):
        self.table = table

        NYT_API_KEY = os.environ['NYT_API_KEY']
        self.link = 'http://api.nytimes.com/svc/search/v2/articlesearch.json'
        self.payload = {'api-key': NYT_API_KEY}
        self._set_filters()

    def _set_filters(self):
        self.payload['fl']='web_url, source, headline, document_type, news_desk, _id, byline'
        self.payload['fq']='source:(The New York Times) AND news_desk:(Foreign, Editorial, Environment, Opinion, Politics, OpEd, U.S., Washington, World)'

    def _set_date(self,date):
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


            for meta in meta_data_list:
                try:
                    #ipdb.set_trace()
                    link = meta['web_url']
                    html = requests.get(link).content
                    soup = BeautifulSoup(html, 'html.parser')
                    article_content = '\n'.join([i.text for i in soup.select('p.story-body-text')])
                    if not article_content:
                        continue
                    meta['content'] = article_content
                    table.insert_one(meta)
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


    def collect_data(self, days):
        today = dt.datetime(2017, 11, 3)

        for day in range(days):
            today -= dt.timedelta(days=1)
            print("Date: {}".format(today))
            self._set_date(today)
            content = self._query()
            hits = content['response']['meta']['hits']
            print("hits: {}".format(hits))
            total_pages = int(hits / 10) + 1
            self._page_loop(total_pages)


if __name__ == '__main__':
    table = open_database_collection()
    scraper=NYTScraper(table)
    scraper.collect_data(days=30)




    # buffer
