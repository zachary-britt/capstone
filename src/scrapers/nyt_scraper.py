'''I feel like I stole a lot of this from either Eric or the DSI solutions'''

import json, os, ipdb
from pymongo import MongoClient
import datetime as dt
import urllib.parse
import time
import scrape_tools as st
import numpy as np
import requests
import bs4

def nyt_parser(link):
    html = requests.get(link).content
    soup = BeautifulSoup(html, 'html.parser')
    article_content = '\n'.join([i.text for i in soup.select('p.story-body-text')])
    return article_content

class NYTScraper:
    def __init__(self):
        self.i=0
        self.table = st.open_database_collection('nyt')

        docs = st.table_to_list(self.table)
        self.seen_urls = { doc['web_url'] for doc in docs }

        self.link = 'http://api.nytimes.com/svc/search/v2/articlesearch.json'
        NYT_API_KEY = os.environ['NYT_API_KEY']
        self.payload = {'api-key': NYT_API_KEY}
        self._set_filters()

    def _set_filters(self):
        self.payload['fl']='web_url, type_of_material, _id, news_desk, headline, pub_date'
        fq = ' AND '.join([
            'source:(The New York Times)',
            'news_desk:(Politics, U.S., Washington, National)',
            'document_type:(article)'
        ])
        self.payload['fq']= fq


    def _set_date_filter(self,begin_date, end_date):
        self.payload['begin_date'] = str(begin_date).replace('-','')
        self.payload['end_date'] = str(end_date).replace('-','')


    def _content_scraper(self, doc):
        link = doc['web_url']
        try:
            soup = st.souper(link, False)
        except:
            print('souping error')
            return

        try:
            art_body = soup.find('article',{'id':'story'})
            art_parts = art_body.find_all('div',{'class':'story-body-supplemental'})
        except:
            print('parting error')
            return

        try:
            children = []
            for part in art_parts:
                children.extend(part.children)

            content_parts = []
            for child in children:
                if type(child) == bs4.element.Tag:
                    content_parts.extend([i.text for i in child.select('p')])

            article_content = '\n'.join(content_parts)

            if not article_content:
                return
        except:
            print('text collection error')
            return

        doc['content'] = article_content
        self.table.insert_one(doc)
        print(self.i, ':', article_content[:80])
        time.sleep(np.random.random()/2+0.3)


    def _page_loop(self, total_pages):
        for page in range(total_pages):
            self.payload['page'] = str(page)
            time.sleep(1)
            response = self._query()

            if response == None:
                continue

            meta_data_list = response['response']['docs']

            for doc in meta_data_list:
                link = doc['web_url']
                self.i += 1
                if link in self.seen_urls:
                    continue
                self._content_scraper(doc)
                self.seen_urls.add(link)

            time.sleep(1)


    def _query(self, second_try=False):
        response = requests.get(self.link, params=self.payload)
        status_code = response.status_code
        if status_code != 200:
            print ('WARNING', status_code)

            # too many requests handling
            if status_code == 429:
                wait_time = 30
                if second_try: wait_time*= 2
                print('Too many requests, waiting {} seconds'.format(wait_time))
                time.sleep(wait_time)

                if not second_try:
                    return self._query(second_try=True)
                else:
                    return None
            return None

        return response.json()


    def collect_data(self, min_date, max_date):
        self._set_date_filter(min_date, max_date)
        content = self._query()
        hits = content['response']['meta']['hits']
        print("hits: {}".format(hits))
        total_pages = int(hits / 10) + 1
        self._page_loop(total_pages)


if __name__ == '__main__':
    scraper=NYTScraper()
    max_date = dt.datetime(2017, 11, 28)
    min_date = dt.datetime(2016, 11, 1)
    scraper.collect_data(min_date, max_date)
