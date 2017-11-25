import scrape_tools as st
from pymongo import MongoClient
import pymongo
import multiprocessing as mp
import urllib.parse




def open_mongo_client():
    username = urllib.parse.quote_plus('admin')
    password = urllib.parse.quote_plus('admin123')
    client = MongoClient('mongodb://%s:%s@127.0.0.1' % (username, password), connect=False)
    return client

def open_database_collection(name):
    client = open_mongo_client()
    db = client['articles']
    table = db[name]
    return table


def process_cursor(cursor):
    target_table = open_database_collection('political_reddit_2')

    fil =   {'$and':[
                {'subreddit': {'$in': ['hillaryclinton', 'politics','progressive', 'The_Donald', 'Conservative','Republican']}},
                {'score': {'$gte': 10} }
            ]}
    proj =  {
            'body': 1,
            'subreddit': 1,
            'author': 1,
            'score': 1,
            'created_utc': 1
            }



    pass

if __name__ == '__main__':

    # table = st.open_database_collection('reddit')
    # table2 = st.open_database_collection('political_reddit')
    #
    # left_subs = ['hillaryclinton', 'politics','progressive']
    # right_subs = ['the_donald', 'conservative','republican']
    #
    # subs = ['hillaryclinton', 'politics','progressive', 'the_donald', 'conservative','republican']


    # mongo db command to create collection of comments from these subs and with a score of at least 5

    root_client = open_mongo_client()
    collection = root_client['news']['reddit']
    n_workers = 6

    cursors = collection.parallel_scan(n_workers)
    processors = [mp.Process(target=process_cursor, args=(cursor,))
                    for cursor in cursors]

    for process in processors:
        process.start()

    for process in processors:
        process.join()

    # chunk_size = 256
    #
    # queue = mp.Queue()
    # func = find_and_copy_political
    # processes = [mp.Process(target=func), args=(queue,) for _ in range(n_workers)]
    # pool = mp.Pool(n_workers)
    # docs = table_cur.find(no_cursor_timeout=True)
    # # doc_count = table_cur.count()
    # pool.map(func, (i, doc) for i,doc in enumerate(docs), chunksize=chunk_size)
    # pool.close()
    # pool.join()

    # 'hillaryclinton', 'politics','progressive',

    '''

    var fil =
    {
        $and:[
                    {subreddit: {$in: [ 'The_Donald', 'Conservative','Republican']}},
                    {score: {$gte: 10} }
             ]
    }

    var bulkInsert = db.political_reddit.initializeUnorderedBulkOp()
    var x = 10000
    var counter = 0

    var proj =
    {
        body: 1,
        subreddit: 1,
        author: 1,
        score: 1,
        created_utc: 1
    }

    db.reddit.find(fil, proj).forEach(
        function(doc){
            bulkInsert.insert(doc);
            counter ++
            if( counter % x == 0){
                bulkInsert.execute()
                bulkInsert = db.political_reddit.initializeUnorderedBulkOp()
          }
        }
      )
    bulkInsert.execute()




        function copyDocuments(sourceCollection, targetCollection, filter){
            var bulkInsert = targetCollection.initializeUnorderedBulkOp();
            sourceCollection.find(filter).forEach(
                function(doc) { bulkInsert.insert(doc); }
            )
            bulkInsert.execute();
        }

        function copy5(sourceCollection, targetCollection, filter){
            var bulkInsert = targetCollection.initializeUnorderedBulkOp();
            sourceCollection.find(filter).limit(5).forEach(
                function(doc) { bulkInsert.insert(doc); }
            )
            bulkInsert.execute();
        }

        var fil =
        {
            $and:[
                        {subreddit: {$in: ['hillaryclinton', 'politics','progressive', 'The_Donald', 'Conservative','Republican']}},
                        {score: {$gte: 5} }
                 ]
        }

        copyDocuments(db.reddit, db.political_reddit, x)
    '''

    table = st.open_database_collection('political_reddit')
