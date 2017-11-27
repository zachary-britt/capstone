import spacy
import pandas as pd
import numpy as np
from spacy.util import minibatch, compounding
from spacy import util
from pathlib import Path
import plac
from pprint import pprint
import ipdb
from collections import Counter
from multiprocessing import Pool

import os
DATA_PATH = os.environ['DATA_PATH']

import zutils
import eval_utils

class Model:

    def __init__(self, model_name, reset_model=False, labels=['left','right']):
        model_dir = DATA_PATH + 'model_cache/' + model_name
        self.model_dir = Path(model_dir)
        self.labels = labels
        self.open_nlp_with_text_cat_(reset_model)


    def open_nlp_with_text_cat_(self, reset_model):
        if not self.model_dir.exists():
            print('No model found at {}'.format(self.model_dir))
            reset_model = True
        if reset_model:
            print('Loading fresh nlp from en_core_web_lg')
            self.nlp = spacy.load('en_core_web_lg')
            self.textcat = self.nlp.create_pipe('textcat')
            self.nlp.add_pipe(self.textcat, last=True)
            for label in self.labels:
                self.textcat.add_label(label)
        else:
            print('Loading pre-trained nlp from {}'.format(self.model_dir))
            self.nlp = spacy.load(self.model_dir)
            self.textcat = self.nlp.get_pipe('textcat')

        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != 'textcat']
        self.nlp.disable_pipes(*other_pipes)  # only train textcat


    def load_and_configure_training_data(self, data_name, **kwargs):

        if kwargs.get('train_all',0):
            test_size=0
        else:
            test_size=0.2

        if kwargs.get('float_bias',0):
            label_type = 'catbias'
        else:
            label_type = 'cats'


        #TODO: push all this kwarg option crap down into zutils
        data = zutils.load_and_configure_data(data_name=data_name,
                                    label_type=label_type, verbose=kwargs.get('verbose',0),
                                    test_data=kwargs.get('test_only',0), test_size=test_size,
                                    labels=['left','right'], get_dates=kwargs.get('dates',0),
                                    space_zip=True, resampling=kwargs.get('resampling',0))

        return data


    def fit(self, data_name, **kwargs):

        data = self.load_and_configure_training_data(data_name, **kwargs)

        train_data = data.get('train')
        val_data = data.get('test', 0)

        self.optimizer = self.nlp.begin_training(n_workers = 8)
        print("Training the model...")

        for i in range(kwargs.get('epochs', 1)):
            seen = 0
            bar = zutils.ProgressBar('Epoch: {}'.format(i+1), len(train_data))
            losses = {}
            # batch up the examples using spaCy's minibatch

            minb = kwargs.get('minb',32.)
            maxb = kwargs.get('maxb',64.)

            batches = minibatch(train_data, size=compounding(minb, maxb, 1.001))
            for j,batch in enumerate(batches):
                texts, labels = zip(*batch)

                self.nlp.update(texts, labels, sgd=self.optimizer, drop=kwargs.get('dropout',0.5),
                                losses=losses)

                seen += len(texts)
                loss_str = 'Avg Loss: {0:.3f}'.format(100*losses['textcat'] / seen)
                bar.progress(seen, loss_str)

            bar.kill(loss_str)
            # end of epoch report
            if val_data.shape[0]: #check val data not empty
                with self.textcat.model.use_params(self.optimizer.averages):
                    scores = self.evaluate_confusion(val_data)


    def score_texts(self, texts):
        scores = []
        docs = (self.nlp.tokenizer(text) for text in texts)
        N = len(texts)
        s = 64
        splits = [s for _ in range(int(N/s))]; splits.append(N % s);
        doc_chunks = ([next(docs) for _ in range(split)] for split in splits)
        bar = zutils.ProgressBar('Eval: ', N)
        for chunk in doc_chunks:
            for doc in self.textcat.pipe(chunk):
                scores.append(doc.cats)
                bar.increment()
        bar.kill()
        return scores

    def evaluate_confusion(self, test_data):
        thresholds = {'left':0.5, 'right':0.5}
        texts, label_dicts = zip(*test_data)
        cat_dicts = [label_dict['cats'] for label_dict in label_dicts]
        scores = self.score_texts(texts)

        for label in thresholds:
            label_scores = np.array([score[label] for score in scores])
            real_labels = np.array([label_dict[label] for label_dict in cat_dicts])
            thresh = thresholds[label]

            M = eval_utils.build_confusion(real_labels, label_scores, thresh)
            eval_utils.print_confusion_report(M, label)

    def predict_proba(self, texts, label=None):
        if label:
            labels = [label]
        else:
            labels = self.labels
        scores = self.score_texts(texts)
        probs = {[ score[label] for score in scores ] for label in labels}
        return np.array(probs)


    def save(self, out_name=None):
        if out_name:
            output_dir = DATA_PATH + 'model_cache/' + out_name
            print('saving to:')
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
        else:
            output_dir = self.model_dir
            print('No output directory given, saving to model directory:')
        print(output_dir)
        self.nlp.to_disk(output_dir)


@plac.annotations(
    data_name=("Dataframe name", "option", 'd', str),
    model_name=("Where to find the model", "option", 'm', str),
    out_name=("where to save the model", 'option', 'o', str),
    reset_model=("Reset model found in model_loc", "flag", "r", bool),
    evaluate_only=('Dont train on data, just evaluate', 'flag','ev', str),
    train_all=('Dont split data, train on full set', 'flag', 'tr', bool),
    resampling=('Type of resampling to use [over, under, none]', 'option', 'rs', str),
    dropout=("Dropout rate to use", 'option', 'do', float),
    min_batch_size=("Minimum Batch size", 'option', "minb", float),
    max_batch_size=("Maximum Batch size", 'option', "maxb", float),
    float_bias=('Use float proportional bias', 'flag', 'fb', bool),
    epochs=("Training epochs", 'option', 'ep', int),
    quiet=('Dont print all over everything','flag','q', bool)
)
def main(   data_name='articles.pkl',
            model_name='spacy_clf',
            out_name=None,
            reset_model=False,
            evaluate_only=False,
            train_all=False,
            resampling='over',
            dropout=0.5,
            min_batch_size=4.,
            max_batch_size=64.,
            float_bias=False,
            epochs=1,
            quiet=False):

    #ipdb.set_trace()

    kwargs = {'evaluate_only':evaluate_only,'train_all':train_all,'resampling':resampling,
                'dropout':dropout, 'epochs':epochs, 'verbose': not quiet,
                'minb':min_batch_size,'maxb':max_batch_size, 'float_bias':float_bias}

    model = Model(model_name, reset_model)

    if not kwargs.get('evaluate_only', 0):
        model.fit(data_name, **kwargs)
        model.save(out_name)
    else:
        model.evaluate_confusion(data_name, **kwargs)


if __name__ == '__main__':
    plac.call(main)
