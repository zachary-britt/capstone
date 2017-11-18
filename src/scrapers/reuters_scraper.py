from scrape_tools import open_database_collection, souper, table_grabber, soup_browser
import threading, multiprocessing
import requests
from bs4 import BeautifulSoup as bs

def build_reuters_search_links():
    links=[]
    for page in range(1,711)

        links.append('https://www.reuters.com/news/archive/politicsNews?view=page&page={}&pageSize=10'.format(page))
    return links

def meta_scraper(table, links):
    for i,link in enumerate(links):
        thread = threading.Thread(name=i, target=meta_scraper_thread, args=(table, link))
        thread.start()
        time.sleep(1)


def meta_scraper_thread(table, link):
    r = requests(link).content
    soup = bs(r, 'html.parse')
    body_soup = soup.find('div', {'class':'news-headline-list'})
    arts = body_soup.find_all('article')
    for art in arts:
        content = art.find('div',{'class':'story-content'})
        a = content.find('a')
        doc = {}
        doc_link = 'https://www.reuters.com'+a['href']
        title = a.get_text
        date = content.find('span',{'class':'timestamp'}).get_text
        doc['title'] = title
        doc['link'] = doc_link
        doc['date'] = date


def content_adder(table, link):
    #find( 'div' {'class':'class'="TwoColumnLayout_container_385r0 Article_content_3kpRX TwoColumnLayout_fluid-left_3DYLH"})
    pass

if __name__ == '__main__':

    table = open_database_collection('articles_reuters')
    links = build_reuters_search_links()

    find( 'div' {'class':'class'="TwoColumnLayout_container_385r0 Article_content_3kpRX TwoColumnLayout_fluid-left_3DYLH"})
