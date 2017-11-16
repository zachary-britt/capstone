import requests, os, ipdb
from pymongo import MongoClient
# from pymongo.errors import DuplicateKeyError, CollectionInvalid
import urllib.parse
from bs4 import BeautifulSoup
import time
from selenium import webdriver
import codecs

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


def soup_browser(browser):
    html = browser.page_source
    soup = BeautifulSoup(html, 'html.parser')
    return soup


def souper(url, on_browser=True, saved=False):
    '''
    convert the stuff at path into Beautiful HTML Soup

    INPUT:
        str:    url         URL or Filepath
        bool:   on_browser  request via browser to wait for javascript injection
        bool:   saved       saved html on local filesystem, (interpret the url as
                            a path)
    OUTPUT:
        Beautiful HTML Soup
    '''

    if saved:
        f = codecs.open(url, 'r')
        html = f.read()
    elif on_browser:
        browser = webdriver.PhantomJS()
        browser.get(url)
        html = browser.page_source
    else:
        try:
            html = requests.get(url).text
        except:
            return None
    soup = BeautifulSoup(html, 'html.parser')
    return soup


def table_grabber(table, field=None):
    '''
    Create a generator over the mongo table which returns the document _id and
    the requested field
    '''
    if field:
        cur = table.find(projection={field:True})
    else:
        cur = table.find()
    while cur.alive:
        yield cur.next()
