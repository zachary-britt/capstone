import requests, os, ipdb
from pymongo import MongoClient
import urllib.parse
from bs4 import BeautifulSoup
import time
from selenium import webdriver
import codecs

import pandas as pd

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException




def open_mongo_client():
    username = urllib.parse.quote_plus('admin')
    password = urllib.parse.quote_plus('admin123')
    client = MongoClient('mongodb://%s:%s@127.0.0.1' % (username, password))
    return client


def open_database_collection(name):
    client = open_mongo_client()
    db = client['articles']
    table = db[name]
    return table


def soup_browser(browser):
    html = browser.page_source
    soup = BeautifulSoup(html, 'html.parser')
    return soup


def souper(url, on_browser=True, saved=False, x_ind=None):
    '''
    convert the stuff at path into Beautiful HTML Soup

    INPUT:
        str:    url         URL or Filepath
        bool:   on_browser  request via browser to wait for javascript injection
        bool:   saved       saved html on local filesystem, (interpret the url as
                            a path)
    OUTPUT:
        Beautiful HTML Soup, or None if Error
    '''

    if saved:
        f = codecs.open(url, 'r')
        html = f.read()
    elif on_browser:
        try:
            browser = webdriver.PhantomJS()

            if x_ind:
                browser.get(url)
                delay = 1 # seconds
                try:
                    myElem = WebDriverWait(browser, delay).until(EC.presence_of_element_located((By.XPATH, x_ind)))
                except TimeoutException:
                    print ("Loading Timeout")

            html = browser.page_source
        except:
            print("Failed to load with browser")
            return None
    else:
        try:
            response = requests.get(url)
            if response.status_code != 200:
                print ('WARNING', response.status_code)
                return None
            else:
                html = response.text
        except:
            print("Failed to load with requests")
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

def table_to_list(table):
    if type(table)==str:
        table = open_database_collection(table)
    gen = table_grabber(table)
    return list(gen)

def open_as_df(table_name):
    table = open_database_collection(table_name)
    t_list = table_to_list(table)
    df = pd.DataFrame(t_list)
    return df
