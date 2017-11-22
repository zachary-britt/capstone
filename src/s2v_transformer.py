'''
Highly borrowed from explosion's sense2vec
'''


from gensim.models import Word2Vec
import spacy

from joblib import Parallel, delayed
import logging
import ujson as json
import plac
import io
import os
from os import path

import numpy as np
import pandas as pd

from multiprocessing import Process
from toolz import partition_all as partition

import ipdb
import re


LABELS = {
    'ENT': 'ENT',
    'PERSON': 'ENT',
    'NORP': 'ENT',
    'FAC': 'ENT',
    'ORG': 'ENT',
    'GPE': 'ENT',
    'LOC': 'ENT',
    'LAW': 'ENT',
    'PRODUCT': 'ENT',
    'EVENT': 'ENT',
    'WORK_OF_ART': 'ENT',
    'LANGUAGE': 'ENT',
    'DATE': 'DATE',
    'TIME': 'TIME',
    'PERCENT': 'PERCENT',
    'MONEY': 'MONEY',
    'QUANTITY': 'QUANTITY',
    'ORDINAL': 'ORDINAL',
    'CARDINAL': 'CARDINAL'
}

pre_format_re = re.compile(r'^[\`\*\~]')
post_format_re = re.compile(r'[\`\*\~]$')
url_re = re.compile(r'\[([^]]+)\]\(%%URL\)')
link_re = re.compile(r'\[([^]]+)\]\(https?://[^\)]+\)')
def strip_meta(text):
    text = link_re.sub(r'\1', text)
    text = text.replace('&gt;', '>').replace('&lt;', '<')
    text = pre_format_re.sub('', text)
    text = post_format_re.sub('', text)
    return text


def annotate_text(batch_id, input_, out_dir):
    out_loc = path.join(out_dir, '%d.txt' % batch_id)
    if path.exists(out_loc):
        return None
    print('Batch', batch_id)
    nlp = spacy.load('en', matcher = None)
    with io.open(out_loc, 'w', encoding='utf8') as file_:
        for record in input_:
            bias, text, date = record
            text = transform_doc(nlp(strip_meta(text)))
            d = {'bias':bias, 'content':text, 'date':date}
            json.dump(d, file_, ensure_ascii=False)

def transform_doc(doc):
    for ent in doc.ents:
        ent.merge()
    try:
        for nc in doc.noun_chunks:
            while len(nc) > 1 and nc[0].dep_ not in ('advmod', 'amod', 'compound'):
                nc = nc[1:]
            nc.merge()
    except:
        pass

    strings = []
    for sent in doc.sents:
        if sent.text.strip():
            strings.append(' '.join(represent_word(w) for w in sent if not w.is_space))
    if strings:
        return '\n'.join(strings)
    else:
        return ''


def represent_word(word):
    if word.like_url:
        return '%%URL|X'
    text = re.sub(r'\s', '_', word.text.lower())
    tag = LABELS.get(word.ent_type_, word.pos_)
    if not tag:
        tag = '?'
    return text + '|' + tag



def main(   in_loc='/home/zachary/dsi/capstone/data/formatted_arts.json',
            out_dir='/home/zachary/dsi/capstone/data/annotated_arts',
            n_workers=4,
            rebuild=False
        ):
    from s2v_transformer import annotate_text
    df = pd.read_json(in_loc, orient='records')
    arr = df.values

    chunk_size = 1000
    chunk = partition(chunk_size, arr)
    func = annotate_text

    parallelize(func, enumerate(chunk), n_workers, [out_dir])



#at the bottom because atom can't parse this right
def parallelize(func, iterator, n_jobs, extra_args):
    extra_args = tuple(extra_args)
    return Parallel(n_jobs=n_jobs)(delayed(func)(*(item + extra_args)) for item in iterator)


if __name__ == '__main__':
    plac.call(main)
