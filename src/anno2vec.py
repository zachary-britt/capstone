from __future__ import print_function, unicode_literals, division
import io
import bz2
import logging
from os import path
import os
import random
from collections import defaultdict
import ipdb
import plac
try:
    import ujson as json
except ImportError:
    import json
from gensim.models import Word2Vec
from preshed.counter import PreshCounter
from spacy.strings import hash_string
import ipdb
import spacy
logger = logging.getLogger(__name__)


class Corpus:
    def __init__(self, directory, min_freq=10):
        self.directory = directory
        self.counts = PreshCounter()
        self.strings = {}
        self.min_freq = min_freq
        self.nlp = spacy.load('en')
    def count_doc(self, words):

        # Get counts for this document
        doc_counts = PreshCounter()
        doc_strings = {}
        for word in words:
            key = hash_string(word)
            doc_counts.inc(key, 1)
            doc_strings[key] = word

        n = 0
        for key, count in doc_counts:
            self.counts.inc(key, count)
            # TODO: Why doesn't inc return this? =/
            corpus_count = self.counts[key]
            # Remember the string when we exceed min count
            if corpus_count >= self.min_freq:  #and (corpus_count - count) < self.min_freq:
                 self.strings[key] = doc_strings[key]
            n += count
        return n

    def add_rare(self, words):

        print("Breaking down rare phrases")


        known_word_keys = set(self.strings.keys())
        new_word_list = []
        for word in words:
            word_key = hash_string(word)
            if not word_key in known_word_keys:
                word = word[:word.find('|')]
                new_words = word.replace('_',' ')
                doc = self.nlp(new_words)
                new_anno_words = [ '{}|{}'.format(n_word.text,n_word.pos_) for n_word in doc ]
                new_word_list.extend(new_anno_words)

        print("Broke into {} new words".format(len(new_word_list)))

        ipdb.set_trace()


        # Get counts for new_words
        doc_counts = PreshCounter()
        doc_strings = {}
        for word in new_word_list:
            key = hash_string(word)
            doc_counts.inc(key, 1)
            doc_strings[key] = word

        m = 0
        forlorn_words={}
        for key, count in doc_counts:
            self.counts.inc(key, count)
            # TODO: Why doesn't inc return this? =/
            corpus_count = self.counts[key]
            # Remember the string when we exceed min count
            if corpus_count >= self.min_freq:
                 self.strings[key] = doc_strings[key]
            else:
                forlorn_words[key] = doc_strings[key]
                m += corpus_count

        print('Could not find a home for {} forlorn words, relabelling'.format(m))

        doc_counts = PreshCounter()
        doc_strings = {}
        for word in forlorn_words.items():
            word = 'RARE'+ word[word.find('|'):]
            key = hash_string(word)
            doc_counts.inc(key, 1)
            doc_strings[key] = word

        for key, count in doc_counts:
            self.counts.inc(key, count)
            corpus_count = self.counts[key]
            #add the word regardless
            self.strings[key] = doc_strings[key]



    def __iter__(self):
        for text_loc in iter_dir(self.directory):
            with io.open(text_loc, 'r', encoding='utf8') as file_:
                sent_strs = list(file_)
                random.shuffle(sent_strs)
                for sent_str in sent_strs:
                    yield sent_str.split()


def iter_dir(loc):
    for fn in os.listdir(loc):
        if path.isdir(path.join(loc, fn)):
            for sub in os.listdir(path.join(loc, fn)):
                yield path.join(loc, fn, sub)
        else:
            yield path.join(loc, fn)

@plac.annotations(
    in_dir=("Location of input directory"),
    out_loc=("Location of output file"),
    n_workers=("Number of workers", "option", "n", int),
    vec_size=("Dimension of the word vectors", "option", "d", int),
    window=("Context window size", "option", "w", int),
    min_count=("Min count", "option", "m", int),
    negative=("Number of negative samples", "option", "g", int),
    nr_iter=("Number of iterations", "option", "i", int),
)
def main(in_dir, out_loc, negative=5, n_workers=4, window=5, vec_size=128, min_count=10, nr_iter=2):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = Word2Vec(
        size=vec_size,
        window=window,
        min_count=min_count,
        workers=n_workers,
        sample=1e-5,
        negative=negative
    )

    corpus = Corpus(in_dir, min_freq=min_count)
    total_words = 0
    total_sents = 0
    for text_no, text_loc in enumerate(iter_dir(corpus.directory)):
        with io.open(text_loc, 'r', encoding='utf8') as file_:
            text = file_.read()
        total_sents += text.count('\n')
        total_words += corpus.count_doc(text.split())
        logger.info("PROGRESS: at batch #%i, processed %i words, keeping %i word types",
                    text_no, total_words, len(corpus.strings))

    #rare processing
    for text_no, text_loc in enumerate(iter_dir(corpus.directory)):
        with io.open(text_loc, 'r', encoding='utf8') as file_:
            text = file_.read()
        corpus.add_rare(text.split())


    model.corpus_count = total_sents
    model.raw_vocab = defaultdict(int)
    for key, string in corpus.strings.items():
        model.raw_vocab[string] = corpus.counts[key]
    model.scale_vocab()
    model.finalize_vocab()
    model.iter = nr_iter

    model.train(corpus, total_examples=model.corpus_count, epochs=model.iter)

    model.save(out_loc)


if __name__ == '__main__':
    plac.call(main)
