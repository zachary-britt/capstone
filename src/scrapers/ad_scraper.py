import ipdb
from pymongo.errors import DuplicateKeyError, CollectionInvalid
from scrape_tools import open_database_collection, souper, table_grabber


def meta_scraper(table):
    # link = 'https://newrepublic.com/political-ad-database'
    soup = souper('../../data/complete.html', saved=True)
    ad_items = soup.find_all('div',{'class' : "campaign-ads-ad"})

    for item in ad_items:
        meta = {}
        meta['link'] = item.find('a', {'class':"campaign-ad-link"} )['href']
        meta['title'] = item.find('h3').text
        meta['supports'] = item.find('h4').text
        meta['date'] = item.find('time')['datetime']
        try:
            table.insert_one(meta)
        except(DuplicateKeyError):
            print('DuplicateKeyError, object already in database:')
            print("-",meta['title'])


def transcript_adder(table):
    gen = table_grabber(table)

    for i,doc in enumerate(gen):
        if not 'content' in doc.keys():
            link = doc['link']
            _id = doc['_id']
            soup = souper(link, on_browser=True)
            content = soup.find('p').text
            table.update_one(filter={'_id':_id}, update={'$set':{'content':content}})
            print(i, content[:70])



if __name__ == '__main__':
    table = open_database_collection('ad_transcripts')
    # meta_scraper(table)
    transcript_adder(table)
