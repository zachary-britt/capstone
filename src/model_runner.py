'''
DEPRECATED
'''


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import anno2vec
from gensim.models import Word2Vec


def load_and_split():
    j_path = '/home/zachary/dsi/capstone/data/annotated_arts/arts.json'
    df = pd.read_json(j_path, orient='records', date_unit='s')

    df_t, df_e = train_test_split(df, test_size=0.2, random_state=100)
    return df_t, df_e


def save_text(df, data_dir):

    full_text = '\n'.join(df.content.values)
    with open(data_dir + 'training_words.txt', 'w') as file_:
        file_.write(full_text)


def vectorize_doc(doc, embedder):
    words = doc.replace('\n',' ').split()

    vecs = [ embedder.wv[word] for word in words  ]
    return vecs

def vectorize_docs(df, embedder):
    df.content.apply( lambda doc: vectorize_doc(doc, embedder) )


if __name__ == '__main__':
    df_t, df_e = load_and_split()

    data_dir = '/home/zachary/dsi/capstone/data/training_words/'
    model_loc = '/home/zachary/dsi/capstone/data/embedder/model'

    if True:
        save_text(df_t, data_dir)
        hypers = { 'negative':10, 'n_workers':4, 'window':6,
                    'vec_size':128, 'min_count':5, 'nr_iter':4 }
        anno2vec.main(data_dir, model_loc, **hypers)

    embedder = Word2Vec.load(model_loc)
