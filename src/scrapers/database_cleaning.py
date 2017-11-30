import scrape_tools as st
import ipdb
from collections import Counter
from pymongo import MongoClient
import pickle
import plac

def remove_dups(table):
    #ipdb.set_trace()
    docs = st.table_to_list(table)

    # urls = [ doc['link'] for doc in docs]
    # _ids = [ doc['_id'] for doc in docs]

    if 'web_url' in docs[0].keys():
        for i,_ in enumerate(docs):
            docs[i]['link'] = docs[i]['web_url']


    pairs = [ (doc['link'],doc['_id']) for doc in docs]
    pair_dict = dict(pairs)
    id_keepers = set(pair_dict.values())
    id_all = {doc['_id'] for doc in docs}

    kill_ids = id_all.difference(id_keepers)

    for _id in kill_ids:
        table.delete_one(filter={'_id':_id})

'''
def remove_contentless(table):
    #do in mongo
    db.fox.deleteMany({content: {$exists: false}})
    pass
'''


def get_political_ids(chunk):
    ''' For off topic text, ensure no political keywords come up
        (Not that I think all of these 'deserve' to be political keywords)
    '''

    substrs = [
        # notable candidates
        # (names mostly last, unless too short/common or well known first name)
        'bush','carson','christie','cruz','fiorina',
        'graham','huckabee','kasich','rand paul','rubio',
        'santorum','perry','walker','malley',

        # notable figures
        'biden','clinton','hillary','obama','bernie','sanders',
        'schumer','pelosi','warren','podesta',

        'trump', 'donald', 'pence',
        'manafort','muller','flinn','kushner', 'tillerson','ivanka', 'ajit pai',
        'paul ryan', 'mccain','mcconnell',

        'president', 'potus',
        'republican','conservative','right wing','gop','libertarian','capitalis',
        'democrat','liberal','left wing','leftist', 'progressive','socialis',
        'congress','senate',
        'd.c.','government',

        # topic terms
        'climate change','global warming','fossil fuel','pruitt','environment','pipeline',
        'evangelical','christian',
        'muslim','islam','arab','syria','iran',
        'jew','isreal',
        'lives matter','blm','racism','racist',
        'social justice','sjw', 'antifa',
        'nationalism', 'nazi','facist', 'alt-right','alt right',
        'net neutrality', 'fcc', 'f.c.c.',
        'tax','economics','economy','globalization','nafta','tpp',
        'immigration','refugees','mexic',
        'crime','police',
        'healthcare', 'insurance',
        'elect ','russia','fraud','conspiracy','vote','fbi','comey','confidential',
        'classified', 'putin','emails',
        'lgbt','gay','lesbian','transgender','trans-','bisexual','gender','rape','sexist',
        'feminis','sexism', 'trans '
        'security','military','industry','corporation'
        ]
    from collections import Counter
    topic_counter = Counter()
    ids = []
    example_text=''
    for doc in chunk:
        text = doc['body'].lower()
        for substr in substrs:
            if substr in text:
                ids.append(doc['_id'])
                example_text = text
                topic_counter[substr] += 1
                break
    print(len(ids))
    return ids, topic_counter

def chunk_cur(cur,size):
    import numpy as np
    sequence = list(cur)
    N = len(sequence)
    slices = np.hstack([np.arange(0, N, size),N])
    for i in range(slices.shape[0]-1):
        yield sequence[slices[i]:slices[i+1]]

def check_political(table, keep_political):
    ''' check if political keywords come up
        for political reddits, ensure political,
        for non-political reddits, cull political
    '''

    from multiprocessing import Pool
    from database_cleaning import get_political_ids
    from collections import Counter
    from functools import reduce
    chunk_gen = chunk_cur(table.find(), 1000)
    #cull_political_process(next(chunk_gen))
    pool = Pool(4)
    ids_lists, topic_counters = zip(*pool.map(get_political_ids, chunk_gen))
    _ids = [ _id for id_list in ids_lists for _id in id_list ]
    #items = [list(topic_counter.items()) for topic_counter in topic_counters]
    topic_counter = reduce(lambda c1,c2: c1+c2, topic_counters)
    #ipdb.set_trace()
    print(len(_ids))
    print(topic_counter)
    if keep_political:
        return table.delete_many( {'_id': {'$nin': _ids}} )
    else:
        #return table.delete_many( {'_id':{'$in': _ids}})
        pass    # already done, except expanded tags for filtering out nonpolitical, politics posts

def main(t_name):
    table = st.open_database_collection(t_name)
    if 'reddit' not in t_name:
        remove_dups(table)
    if 'reddit' in t_name:
        if 'neutral' in t_name:
            return check_political(table, keep_political=False)
        else:
            return check_political(table, keep_political=True)

if __name__ == '__main__':
    cur = plac.call(main)
