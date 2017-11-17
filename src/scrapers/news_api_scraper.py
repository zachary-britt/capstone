'''
DEPRECATED
'''



import requests, json, os, ipdb, sys
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, CollectionInvalid
import datetime as dt
import urllib.parse
from bs4 import BeautifulSoup
import time

'''
Powered by news api!
'''


def open_mongo_client():
    username = urllib.parse.quote_plus('admin')
    password = urllib.parse.quote_plus('admin123')
    client = MongoClient('mongodb://%s:%s@127.0.0.1' % (username, password))
    return client


def open_database_collection(name):
    client = open_mongo_client()
    db = client['news']
    table = db[name]
    return table



def souper(link):
    try:
        html = requests.get(link).content
        soup = BeautifulSoup(html, 'html.parser')
        return soup
    except:
        return None


def wp_parser(soup):
    art_class_tag = 'paywall'
    art_soup = soup.find('article', {'class':art_class_tag})
    if art_soup:
        article_content = '\n'.join([i.text for i in art_soup.select('p')])
        if article_content:
            return article_content
    return None

def hp_parser(soup):
    art_class_tag = 'entry__text js-entry-text bn-entry-text'
    art_soup = soup.find('div', {'class':art_class_tag})
    if art_soup:
        article_content = '\n'.join([i.text.replace('\xa0','') for i in art_soup.select('p')])
        if article_content:
            return article_content
    return None


def nyt_parser(soup):
    article_content = '\n'.join([i.text for i in soup.select('p.story-body-text')])
    return article_content

def reuters_parser(soup):
    art_class_tag = 'ArticleBody_body_2ECha'
    art_soup = soup.find('div', {'class':art_class_tag})
    if art_soup:
        article_content = '\n'.join([i.text for i in art_soup.select('p')])
        if article_content:
            return article_content
    return None

# def cnn_parser(soup):
#     art_class_tag = 'l-container'
#     art_soup = soup.find('div', {'class':art_class_tag})
#     if art_soup:
#         article_content = '\n'.join([p.text for p in art_soup.find_all('div',{'class':'zn-body__paragraph'})])
#         # article_content = '\n'.join([i.text for i in art_soup.select('p')])
#         if article_content:
#             return article_content
#     return None


def breitbart_parser(soup):
    art_class_tag = 'entry-content'
    art_soup = soup.find('div', {'class':art_class_tag})
    if art_soup:
        article_content = '\n'.join([i.text for i in art_soup.select('p')])
        if article_content:
            return article_content
    return None


def bbc_parser(soup):
    art_class_tag = 'story-body__inner'
    art_soup = soup.find('div', {'class':art_class_tag})
    if art_soup:
        article_content = '\n'.join([i.text for i in art_soup.select('p')])
        if article_content:
            return article_content
    return None


def time_parser(soup):
    # art_class_tag = 'padded'
    art_soup = soup.find('div', {'id':"article-body"})
    if art_soup:
        article_content = '\n'.join([i.text for i in art_soup.select('p')])
        if article_content:
            return article_content
    return None


class GeneralScraper:

    _parsers = {'the-washington-post'   : wp_parser,
                'the-huffington-post'   : hp_parser,
                'the-new-york-times'    : nyt_parser,
                'reuters'               : reuters_parser,
                'breitbart-news'        : breitbart_parser,
                'bbc-news'              : bbc_parser,
                'time'                  : time_parser
                }

    def __init__(self, source_name):
        table_name = 'articles_'+source_name.replace('-','_')
        self.table = open_database_collection(table_name)
        API_KEY = os.environ['NEWS_API_KEY']
        self.link = 'https://newsapi.org/v1/articles?'
        self.payload = {'apiKey': API_KEY}
        self.payload['source'] = source_name
        self.payload['sortBy']='top'
        self.parser =  self._parsers[source_name]
        self._article_loop()


    def _article_loop(self):
        j_out = self._query()
        if j_out == None:
            time.sleep(1)
            j_out = self._query()
            if j_out == None:
                return
        meta_data_list = j_out['articles']
        for meta in meta_data_list:
            try:
                #ipdb.set_trace()
                meta.pop('urlToImage')

                #check to ensure meta data includes author, url, and date
                if (meta['author'] == None or meta['url'] == None or
                                meta['publishedAt'] == None):
                    continue
                date = meta.pop('publishedAt')[:10].replace('-','/')
                link = meta['url']
                soup = souper(link)
                if not soup:
                    continue
                article_content = self.parser(soup)
                if not article_content:
                    continue
                meta['content'] = article_content
                self.table.insert_one(meta)
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


if __name__ == '__main__':
    sources = [
        'the-washington-post',
        'the-huffington-post',
        'the-new-york-times',
        'reuters',
        'breitbart-news',
        'bbc-news',
        'time'
    ]

    for i in range(7):
        gs = GeneralScraper(sources[i])





#buffer
