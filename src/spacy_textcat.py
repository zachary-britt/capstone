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

            low_data = self.cfg.get('low_data')
            pipe_cfg = {'low_data', low_data}
            self.textcat = self.nlp.create_pipe('textcat')

            self.nlp.add_pipe(self.textcat, last=True)
            for label in self.labels:
                self.textcat.add_label(label)
        else:
            print('Loading pre-trained nlp from {}'.format(self.model_dir))
            # ipdb.set_trace()
            self.nlp = spacy.load(self.model_dir)
            self.textcat = self.nlp.get_pipe('textcat')

        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != 'textcat']
        self.nlp.disable_pipes(*other_pipes)  # only train textcat


    def load_and_configure_training_data(self, data_name):
        data = zutils.load_and_configure_data(data_name, **self.cfg)
        return data

    def _make_optimizer(self):
        '''
        Allows for custom tweaking of adam optimizer
        '''
        # cannibalized from spacy/spacy/_ml/create_default_optimizer
        optimizer = self.nlp.begin_training(n_workers = 8)
        optimizer.learn_rate = self.cfg.get('learn_rate', 0.001)
        optimizer.beta1 = self.cfg.get('optimizer_B1', 0.9)
        optimizer.beta2 = self.cfg.get('optimizer_B2', 0.999)
        optimizer.eps = self.cfg.get('optimizer_eps', 1e-08)
        optimizer.L2 = self.cfg.get('L2_penalty', 1e-6)
        optimizer.max_grad_norm = self.cfg.get('grad_norm_clip', 1.)
        return optimizer

    def fit(self, data_name, **kwargs):
        '''
        fits the model
        '''
        self.cfg.update(kwargs)

        data = self.load_and_configure_training_data(data_name)
        train_data = data.get('train')
        val_data = data.get('test')

        optimizer = self._make_optimizer()

        print("Training the model...")

        for i in range(self.cfg.get('epochs', 1)):
            total_seen = 0
            n = 0

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
                loss_str = 'Avg Loss: {0:.3f}'.format(100*losses['textcat'] / (j+1))
                bar.progress(total_seen, loss_str)


                ''' run test on subset of validation set so you don't get bored '''
                if len(val_data) and self.cfg.get('verbose'): #check val data not empty
                    n += bs
                    if n >= 10000:
                        n=0
                        t = bar.kill(loss_str)
                        with self.textcat.model.use_params(optimizer.averages):
                            self.in_progress_val(val_data)

                        bar = zutils.ProgressBar('Epoch: {}'.format(i+1), len(train_data))
                        bar.start_time -= t
                # end epoch

            bar.kill(loss_str)

            # end of epoch report
            if len(val_data) and self.cfg.get('verbose'): #check val data not empty
                with self.textcat.model.use_params(optimizer.averages):
                    self.evaluate_confusion(val_data)

    def in_progress_val(self, val_data):
        '''
        subsets val_data to create a quick (in progress) evauation metric
        '''
        # subset validation set
        val_arr = np.array(val_data)
        inds = np.arange(val_arr.shape[0])
        mini_val_inds = np.random.choice(inds, 1000, replace=False)
        mini_val = val_arr[mini_val_inds].tolist()
        self.evaluate_confusion(mini_val)


    def score_texts(self, texts, verbose=True):
        ''' Produces category probability scores for texts
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

    def evaluate_confusion(self, test_data):
        '''
        Creates evauation metrics for test_data.

        test_data can either be in a (text, cats) zip, or
        a dataframe with text in df['content'] and labels in
        df['orient'] or df[['left','right']]
        '''
        thresholds = {'left':0.5, 'right':0.5}

        # tediously handle list or pandas DataFrame input
        if type(test_data) == list:
            texts, label_dicts = zip(*test_data)
            cat_dicts = [label_dict['cats'] for label_dict in label_dicts]
        elif type(test_data) == pd.DataFrame:
            texts = test_data['content'].tolist()
            if 'left' in test_data.columns:
                is_lefts = test_data['left']
                is_rights = test_data['right']
            elif 'orient' in test_data.columns:
                is_lefts = np.where(test_data['orient'].values == 'left',1,0)
                is_rights = np.where(test_data['orient'].values == 'right',1,0)
            else:
                print('test data label column not found')
                return
            cat_dicts = [{'left':z[0],'right':z[1]} for z in zip(is_lefts, is_rights)]
        else:
            print('test_data is of type {}, must be list or dataframe')
            return

        scores_df = self.score_texts(texts)

        for label in thresholds:
            label_scores = scores_df[label]
            # label_scores = np.array([score[label] for score in scores])
            real_labels = np.array([label_dict[label] for label_dict in cat_dicts])
            thresh = thresholds[label]

            M = eval_utils.build_confusion(real_labels, label_scores, thresh)
            eval_utils.print_confusion_report(M, label)
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


    def save(self, out_name=None):
        'Save text categorization model to disk'
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
    min_batch_size=("Minimum Batch size", 'option', "minb", float),
    max_batch_size=("Maximum Batch size", 'option', "maxb", float),
    float_bias=('Use float proportional bias', 'flag', 'fb', bool),
    low_data=('simplified model','flag', 'low_d', bool),
    epochs=("Training epochs", 'option', 'ep', int),
    glove=("use gloVe vectors", 'flag','gl', bool),
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
            L2_penalty=1e-6,
            min_batch_size=4.,
            max_batch_size=16.,
            float_bias=False,
            low_data=False,
            epochs=1,
            glove=False,
            quiet=False
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

    kwargs = {  'out_name':out_name, 'reset':reset, 'test_all':test_all,
                'train_all':train_all,'resampling':resampling,'dropout':dropout,
                'minb':min_batch_size,'maxb':max_batch_size, 'label_type':label_type,
                'epochs':epochs,'verbose': not quiet, 'glove':glove,
                'L2_penalty':L2_penalty, "zipit":True, 'max_class_size': maxN}

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
