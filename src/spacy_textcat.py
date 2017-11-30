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
import pickle

import os
DATA_PATH = os.environ['DATA_PATH']

import zutils
import eval_utils

class Model:

    def __init__(self, model_name, **kwargs):
        self.cfg = kwargs
        self.cfg['catnest']=True
        model_dir = DATA_PATH + 'model_cache/' + model_name
        self.model_dir = Path(model_dir)
        self.labels = kwargs.get('labels',['left', 'right'])
        self.open_nlp_with_text_cat_()


    def open_nlp_with_text_cat_(self):
        if not self.model_dir.exists():
            print('No model found at {}'.format(self.model_dir))
            new_model = True
        else:
            new_model = False
        if new_model or self.cfg.get('reset'):
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


    def load_and_configure_training_data(self, data_name):
        data = zutils.load_and_configure_data(data_name, **self.cfg)
        return data


    def fit(self, data_name, **kwargs):
        self.cfg.update(kwargs)

        data = self.load_and_configure_training_data(data_name)
        train_data = data.get('train')
        val_data = data.get('test')

        self.optimizer = self.nlp.begin_training(n_workers = 8)
        print("Training the model...")

        for i in range(self.cfg.get('epochs', 1)):
            seen = 0;
            n = 0;

            bar = zutils.ProgressBar('Epoch: {}'.format(i+1), len(train_data))
            losses = {}
            # batch up the examples using spaCy's minibatch

            minb = self.cfg.get('minb',32.)
            maxb = self.cfg.get('maxb',64.)

            batches = minibatch(train_data, size=compounding(minb, maxb, 1.001))
            for j,batch in enumerate(batches):
                texts, labels = zip(*batch)

                self.nlp.update(texts, labels, sgd=self.optimizer, drop=self.cfg.get('dropout',0.5),
                                losses=losses)

                seen += len(texts)
                loss_str = 'Avg Loss: {0:.3f}'.format(100*losses['textcat'] / seen)
                bar.progress(seen, loss_str)

                n += len(texts)

                if n >= 10000:
                    n=0
                    t = bar.kill(loss_str)
                    with self.textcat.model.use_params(self.optimizer.averages):
                        val_arr = np.array(val_data)
                        inds = np.arange(val_arr.shape[0])
                        mini_val_inds = np.random.choice(inds, 1000, replace=False)
                        mini_val = val_arr[mini_val_inds].tolist()
                        scores=self.evaluate_confusion(mini_val)
                    bar = zutils.ProgressBar('Epoch: {}'.format(i+1), len(train_data))
                    bar.start_time -= t
            bar.kill(loss_str)
            # end of epoch report
            if len(val_data): #check val data not empty
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
        return scores

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
    data_name=("Dataframe name"),
    model_name=("Where to find the model", "option", 'm', str),
    out_name=("where to save the model", 'option', 'o', str),
    reset=("Reset model found in model_loc", "flag", "rt", bool),
    test_all=('Dont train on data, just evaluate', 'flag','ev', bool),
    train_all=('Dont split data, train on full set', 'flag', 'tr', bool),
    resampling=('Type of resampling to use [over, under, none]', 'option', 'rs', str),
    maxN=('max class size for choose N resampling', 'option', 'maxN', int),
    dropout=("Dropout rate to use", 'option', 'do', float),
    min_batch_size=("Minimum Batch size", 'option', "minb", float),
    max_batch_size=("Maximum Batch size", 'option', "maxb", float),
    float_bias=('Use float proportional bias', 'flag', 'fb', bool),
    epochs=("Training epochs", 'option', 'ep', int),
    quiet=('Dont print all over everything','flag','q', bool)
)
def main(   data_name,
            model_name='spacy_clf',
            out_name=None,
            reset=False,
            test_all=False,
            train_all=False,
            resampling='over',
            maxN=2000,
            dropout=0.6,
            min_batch_size=4.,
            max_batch_size=16.,
            float_bias=False,
            epochs=1,
            quiet=False
            ):

    if float_bias:
        label_type = 'catbias'
    else:
        label_type = 'cats'

    kwargs = {  'out_name':out_name, 'reset':reset, 'test_all':test_all,
                'train_all':train_all,'resampling':resampling,'dropout':dropout,
                'minb':min_batch_size,'maxb':max_batch_size, 'label_type':label_type,
                'epochs':epochs,'verbose': not quiet,
                "zipit":True, 'max_class_size': maxN}

    model = Model(model_name, **kwargs)

    if not kwargs.get('test_all'):
        model.fit(data_name, **kwargs)
        model.save(out_name)
        return None, None
    else:
        data = zutils.load_and_configure_data(data_name, **model.cfg)['test']
        scores = model.evaluate_confusion(data)
        #y_pred = model.predict_proba(zip(*data)[0])
        return scores, data


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    scores, data = plac.call(main)
    if scores:

        text, y_true = zip(*data)
        y_true = [y['cats'] for y in y_true]
        r_true = np.array([y['right'] for y in y_true])
        l_true = np.array([y['left'] for y in y_true])

        r_pred = np.array([score['right'] for score in scores])
        l_pred = np.array([score['left'] for score in scores])

        with open('my_thing.pkl','wb') as f:
            pickle.dump([r_true, l_true, r_pred, l_pred], f)

        eval_utils.make_roc(r_true, r_pred, 'right')
        eval_utils.make_roc(l_true, l_pred, 'left')

        plt.show()


        #
