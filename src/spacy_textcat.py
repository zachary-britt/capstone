import spacy
import pandas as pd
import numpy as np
from spacy.util import minibatch, compounding
from pathlib import Path
import plac
from pprint import pprint
import ipdb
import zutils

import os
DATA_PATH = os.environ['DATA_PATH']


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

        data = zutils.load_and_configure_data(data_name=data_name,
                                    label_type='cats', verbose=kwargs.get('verbose',0),
                                    test_data=kwargs.get('test_only',0), test_size=0.2,
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
            batches = minibatch(train_data, size=compounding(4., 64., 1.001))
            for j,batch in enumerate(batches):
                texts, labels = zip(*batch)

                self.nlp.update(texts, labels, sgd=self.optimizer, drop=kwargs.get('dropout',0.5),
                                losses=losses)

                ''' in process validation so there's something to watch '''
                seen += len(texts)
                loss_str = 'Avg Loss: {0:.3f}'.format(loss['textcat'] / seen)*100)
                bar.progress(seen, loss_str)
            bar.kill(loss_str)
            # end of epoch report
            scores = self.evaluate(val_data)


    def evaluate(self, annotated_data, verbose=True):
        with self.textcat.model.use_params(self.optimizer.averages):
            #ipdb.set_trace()
            texts, labels = zip(*annotated_data)
            docs = (self.nlp.tokenizer(text) for text in texts)
            tp = 1e-8  # True positives
            fp = 1e-8  # False positives
            fn = 1e-8  # False negatives
            tn = 1e-8  # True negatives
            for i, doc in enumerate(self.textcat.pipe(docs)):
                gold = labels[i]['cats']
                for label, score in doc.cats.items():
                    if label not in gold:
                        continue
                    if score >= 0.5 and gold[label] >= 0.5:
                        tp += 1.
                    elif score >= 0.5 and gold[label] < 0.5:
                        fp += 1.
                    elif score < 0.5 and gold[label] < 0.5:
                        tn += 1
                    elif score < 0.5 and gold[label] >= 0.5:
                        fn += 1
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f_score = 2 * (precision * recall) / (precision + recall)

            if verbose:
                print('{:^5}\t{:^5}\t{:^5}'.format('P', 'R', 'F'))
                print('{0:.3f}\t{1:.3f}\t{2:.3f}'.format(precision, recall, f_score))

            return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score}


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
    data_name=("Dataframe name", "option", 'd'),
    model_name=("Where to find the model", "option", 'm'),
    out_name=("where to save the model", 'option', 'o'),
    reset_model=("Reset model found in model_loc", "flag", "r"),
    evaluate_only=('Dont train on data, just evaluate', 'flag','ev'),
    train_all=('Dont split data, train on full set', 'flag', 'tr'),
    resampling=('Type of resampling to use [over, under]', 'option', 's'),
    dropout=("Dropout rate to use", 'option', 'do'),
    epochs=("Training epochs", 'option', 'ep'),
    learning_rate=('NN learning rate', 'option', 'lr'),
    quiet=('Dont print all over everything','flag','q')
)
def main(   data_name='articles.pkl',
            model_name='spacy_clf',
            out_name='spacy_clf',
            reset_model=False,
            evaluate_only=False,
            train_all=False,
            resampling='over',
            dropout=0.5,
            epochs=1,
            learning_rate=0.001,
            quiet=False):

    #ipdb.set_trace()

    kwargs = {'evaluate_only':evaluate_only,'train_all':train_all,'resampling':resampling,
                'dropout':dropout, 'epochs':epochs,'learning_rate':learning_rate,
                'verbose': not quiet}

    model = Model(model_name, reset_model)

    if not kwargs.get('evaluate_only', 0):
        model.fit(data_name, **kwargs)
        model.save(out_name)
    else:
        model.evaluate(data_name, **kwargs)


if __name__ == '__main__':
    plac.call(main)
