'''
Highly borrowed from explosion's sense2vec
'''

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
    #out_loc = path.join(out_dir, '%d.json' % batch_id)
    out_loc = path.join(out_dir, 'arts.json')
    if path.exists(out_loc):
        return None
    print('Batch', batch_id)
    nlp = spacy.load('en', matcher = None)
    output=input_
    for i, record in enumerate(input_):
        text = record[1]
        text = transform_doc(nlp(strip_meta(text)))
        output[i][1] = text
    data = np.array(output)
    df = pd.DataFrame(data, columns=['bias','content','date'])
    df.to_json(out_loc, orient='records', force_ascii=False)

def transform_doc(doc):
    for ent in doc.ents:
        ent.merge(ent_type=ent.root.ent_type, pos=ent.root.pos)

    for nc in doc.noun_chunks:
        while len(nc) > 1 and nc[0].dep_ not in ('advmod', 'amod', 'compound'):
            nc = nc[1:]
        nc.merge(ent_type=nc.root.ent_type, pos=nc.root.pos)

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
            n_workers=2,
            rebuild=False
        ):
    from s2v_transformer import annotate_text
    df = pd.read_json(in_loc, orient='records')
    lst = df.values.tolist()

    chunk_size = 500
    chunk = partition(chunk_size, lst)
    func = annotate_text
    func(0, lst, out_dir)
    #parallelize(func, enumerate(chunk), n_workers, [out_dir])


#at the bottom because atom can't parse this right
def parallelize(func, iterator, n_jobs, extra_args):
    extra_args = tuple(extra_args)
    return Parallel(n_jobs=n_jobs)(delayed(func)(*(item + extra_args)) for item in iterator)


if __name__ == '__main__':
    plac.call(main)
