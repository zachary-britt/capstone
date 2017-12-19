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
import sys
import os
DATA_PATH = os.environ['DATA_PATH']

import zutils
import eval_utils


class Spacecat:
    def __init__(self, **kwargs):
        if not kwargs.get('model_name'):
            print('Must provide a model name')
            sys.exit()

        self.cfg = kwargs

        self.verbose = self.cfg.get('verbose')
        self.super_verbose = self.cfg.get('super_verbose')
        self.cfg['catnest']=True

        model_dir = DATA_PATH + 'model_cache/' + self.cfg.get('model_name')
        self.model_dir = Path(model_dir)

        self.labels = kwargs.get('labels',['left', 'right'])
        self.open_nlp_with_text_cat_()

        if not self.cfg.get('out_name'):
            self.cfg['out_name'] = self.cfg['model_name']


    def open_nlp_with_text_cat_(self):
        '''
        Loads prebuilt model
        -If spacy model, it creates the textcat pipe component and adds it
        -If reloaded from model saved here, loads that model and textcat
        '''
        if not self.model_dir.exists():
            print('No model found at {}'.format(self.model_dir))
            new_model = True
        else:
            new_model = False
        if new_model or self.cfg.get('reset'):

            if self.cfg.get('glove'):
                spacy_model_name = 'en_vectors_web_lg'
            else:
                spacy_model_name = 'en_core_web_lg'

            print('Loading fresh nlp from {}'.format(spacy_model_name))
            self.nlp = spacy.load(spacy_model_name)

            self.textcat = self.nlp.create_pipe('textcat')
            self.textcat.cfg['low_data'] = self.cfg.get('low_data')
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
        '''
        INPUT:
        - data_name: str of data location relative to DATA_PATH
        OUTPUT:
        - data:     dict    {'train': training_data, 'test': testing_data}
                    *_data being a zip of (texts, annotations)
        '''
        rr = self.cfg.get('reddit_ratio')
        if rr:
            zutils.red_dominant(**self.cfg)

        data = zutils.load_and_configure_data(**self.cfg)
        return data

    def _make_optimizer(self):
        '''
        INPUT:
        - None
        OUTPUT:
        - optimizer: thinc.neural.optimizers.Optimizer

        Allows for custom tweaking of adam optimizer
        '''
        # cannibalized from spacy/spacy/_ml/create_default_optimizer

        optimizer = self.nlp.begin_training(n_workers = -1)
        optimizer.learn_rate = self.cfg.get('learn_rate', 0.001)
        optimizer.beta1 = self.cfg.get('optimizer_B1', 0.9)
        optimizer.beta2 = self.cfg.get('optimizer_B2', 0.999)
        optimizer.eps = self.cfg.get('optimizer_eps', 1e-08)
        optimizer.L2 = self.cfg.get('L2_penalty', 1e-6)
        optimizer.max_grad_norm = self.cfg.get('grad_norm_clip', 1.)
        return optimizer

    def fit(self, **kwargs):
        '''
        INPUT
        - kwargs: updates to cfg dict.
        trains the model.
        '''

        self.cfg.update(kwargs)
        check_in_interval = self.cfg.get('check_in_interval', 10000)


        data_name = self.cfg.get('data_name')
        data = self.load_and_configure_training_data(data_name)
        train_data = data.get('train')
        val_data = data.get('test')

        optimizer = self._make_optimizer()

        print(self.textcat.cfg)

        print("Training the model...")

        for i in range(self.cfg.get('epochs', 1)):
            total_seen = 0
            n = 0
            k=1
            bar = zutils.ProgressBar('Epoch: {}'.format(i+1), len(train_data))
            losses = {}

            # increase batch size from minb to maxb. Apparently this is the cool new thing
            minb = self.cfg.get('minb',32.)
            maxb = self.cfg.get('maxb',64.)
            batches = minibatch(train_data, size=compounding(minb, maxb, 1.001))
            for j, batch in enumerate(batches):

                texts, labels = zip(*batch)

                self.nlp.update(texts, labels, sgd=optimizer, drop=self.cfg.get('dropout',0.5),
                                losses=losses)
                bs = len(texts)
                total_seen += bs
                loss_str = 'Avg Loss: {0:.3f}'.format(100*losses['textcat'] / (k))
                bar.progress(total_seen, loss_str)
                k+=1

                ''' run test on subset of validation set so you don't get bored '''
                n += bs
                if n >= check_in_interval:
                    n=0
                    if len(val_data) and self.cfg.get('verbose'): #check val data not empty

                        t = bar.kill(loss_str)
                        print(self.cfg.get('model_name'))
                        with self.textcat.model.use_params(optimizer.averages):
                            self.in_progress_val(val_data, N=total_seen)

                        bar = zutils.ProgressBar('Epoch: {}'.format(i+1), len(train_data))
                        bar.start_time -= t
                    losses={}
                    k=1
            # end epoch
            bar.kill(loss_str)

            # end of epoch report
            if len(val_data) and self.cfg.get('verbose'): #check val data not empty
                with self.textcat.model.use_params(optimizer.averages):
                    self.evaluate_confusion(val_data)

    def in_progress_val(self, val_data, N=None):
        '''
        subsets val_data to create a quick (in progress) evauation metric
        '''
        # subset validation set
        val_arr = np.array(val_data)
        inds = np.arange(val_arr.shape[0])
        mini_val_inds = np.random.choice(inds, 1000, replace=False)
        mini_val = val_arr[mini_val_inds].tolist()
        self.evaluate_confusion(mini_val, N=N, mini_val=True)


    def score_texts(self, texts, verbose=True):
        '''
        Produces category probability scores for texts
        '''
        scores = []
        docs = (self.nlp.tokenizer(text) for text in texts)
        N = len(texts)
        s = 64
        splits = [s for _ in range(int(N/s))]; splits.append(N % s);
        doc_chunks = ([next(docs) for _ in range(split)] for split in splits)
        if verbose: bar = zutils.ProgressBar('Eval: ', N)

        for chunk in doc_chunks:
            for doc in self.textcat.pipe(chunk):
                scores.append(doc.cats)
                if verbose: bar.increment()

        if verbose: bar.kill()

        #to pandas output
        scores_df = pd.DataFrame(scores)

        return scores_df

    def evaluate_confusion(self, test_data, N=None, mini_val=False):
        '''
        Creates evauation metrics for test_data.

        test_data in a (text, cats) zip
        '''

        texts, label_dicts = zip(*test_data)
        cat_dicts = [label_dict['cats'] for label_dict in label_dicts]

        scores_df = self.score_texts(texts)

        if len(self.labels) == 2:
            thresholds = {'left':0.5, 'right':0.5}
            for label in thresholds:

                label_scores = scores_df[label]
                # label_scores = np.array([score[label] for score in scores])
                real_labels = np.array([label_dict[label] for label_dict in cat_dicts])
                thresh = thresholds[label]

                M = eval_utils.build_confusion(real_labels, label_scores, thresh)
                eval_utils.print_confusion_report(M, label)

        else:
            thresholds = {'left':0.05, 'right':0.05}
            label_scores = scores_df['bias']
            true_labels = np.array([label_dict['bias'] for label_dict in cat_dicts])

            t = thresholds['right']

            if mini_val:
                title = "Minival ROC for: {} after {} samples".format(self.cfg.get('out_name'), N)
            else:
                title = "Eval ROC for: {}".format(self.cfg.get('out_name'))

            eval_utils.make_roc(true_labels, label_scores, 'right', title=title)

            M = eval_utils.build_confusion(true_labels, label_scores, t)
            eval_utils.print_confusion_report(M, 'right')

            t = thresholds['left']

            label_scores *= -1
            true_labels *= -1

            out_dir = '../figures/eval_rocs/'
            out = out_dir + self.cfg.get('out_name')
            eval_utils.make_roc(true_labels, label_scores, 'left', title=title, file_path = out)
            # eval_utils.make_roc(true_labels, label_scores, 'left', title=title, action='show')

            M = eval_utils.build_confusion(true_labels, label_scores, t)
            eval_utils.print_confusion_report(M, 'left')

        return scores_df

    def predict_proba(self, df, verbose=True):
        '''
        Sklean style predict_proba
        Expects df to have text in 'content' field
        '''
        texts = df['content'].tolist()
        labels = self.labels
        scores = self.score_texts(texts, verbose)
        probs = {[ score[label] for score in scores ] for label in labels}
        return np.array(probs)


    def single_query(self, text):
        '''
        Returns scores for a single model. More compartmentalized than externally
        pulling up the nlp attribute
        '''
        labels = self.labels
        doc = self.nlp(text)
        scores = doc.cats
        return scores


    def save(self, out_name=None):
        'Save model to disk'
        if out_name:
            output_dir = DATA_PATH + 'model_cache/' + out_name
            print('saving to:')
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
        elif self.cfg.get('out_name'):
            output_dir = DATA_PATH + 'model_cache/' + self.cfg.get('out_name')
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
    resampling=('Type of resampling to use [over, under, choose_n, none]', 'option', 'rs', str),
    maxN=('max class size for choose N resampling', 'option', 'maxN', int),
    dropout=("Dropout rate to use", 'option', 'do', float),
    L2_penalty=("L2 regularization param", 'option', 'l2', float),
    learn_rate=('Learning rate used by optimizer','option','lr', float),
    minb=("Minimum Batch size", 'option', "minb", float),
    maxb=("Maximum Batch size", 'option', "maxb", float),
    float_bias=('Use float proportional bias', 'flag', 'fb', bool),
    low_data=('simplified model','flag', 'low_d', bool),
    epochs=("Training epochs", 'option', 'ep', int),
    glove=("use gloVe vectors", 'flag','gl', bool),
    reddit_ratio=('ratio of reddit to arts', 'option', 'rr', float),
    test_cap=('maximum test size', 'option', 'tc', int),
    tanh_setup=('Single label tanh config','flag','ts',bool),
    quiet=('Dont print all over everything','flag','q', bool),
    super_verbose=('Print all of everything', 'flag', 'sv', bool),
    check_in_interval=('Interval at which to do mini-val', 'option', 'cii', int),
    model_getter=('Return model', 'flag', 'mg', bool)
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
            L2_penalty=1e-6,
            learn_rate=0.001,
            minb=4.,
            maxb=32.,
            float_bias=False,
            low_data=False,
            epochs=1,
            glove=False,
            reddit_ratio=0.0,
            test_cap=0,
            tanh_setup=False,
            quiet=False,
            super_verbose=False,
            check_in_interval=50000,
            model_getter=False
            ):
    '''
    Builds text categorization model with spacy
    Loads model from model_name or creates new one
    Trains, train/tests, or tests the model on provided 'data_name'
    Saves model to out_name or model_name if not provided
    '''
    if float_bias:
        label_type = 'catbias'
    else:
        label_type = 'cats'

    zipit=True
    verbose = not quiet
    max_class_size = maxN

    if tanh_setup:
        labels = ['bias']
        label_type = 'tbias'

    kwargs = dict(locals())

    spacecat = Spacecat(**kwargs)

    if model_getter:
        return spacecat

    if not kwargs.get('test_all'):
        spacecat.fit(**kwargs)
        spacecat.save(out_name)
    else:
        data = zutils.load_and_configure_data(**spacecat.cfg)['test']
        spacecat.evaluate_confusion(data)


if __name__ == '__main__':
    plac.call(main)
