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


    def load_and_configure_training_data(self, data_loc, verbose):

        train_data, val_data = zutils.load_and_configure_data(data_name='articles.pkl',
                                    label_type='cats', verbose=verbose, test_data=False,
                                    test_size=0.2, labels=['left','right'],
                                    get_dates=False, space_zip=True, resampling='over')

        return train_data, val_data


    def fit(self, data_loc, n_epochs, verbose=True):

        train_data, val_data= self.load_and_configure_training_data(data_loc, verbose)

        self.optimizer = self.nlp.begin_training(n_workers = 6)
        print("Training the model...")

        for i in range(n_epochs):
            seen = 0
            bar = zutils.ProgressBar('Epoch: {}'.format(i+1), len(train_data))
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4., 64., 1.01))
            for j,batch in enumerate(batches):
                texts, labels = zip(*batch)

                self.nlp.update(texts, labels, sgd=self.optimizer, drop=0.4, losses=losses)

                ''' in process validation so there's something to watch '''
                seen += len(texts)
                loss_str = 'Avg Loss: {0:.3f}'.format(losses['textcat']/seen)
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
        output_dir = DATA_PATH + 'model_cache/' + out_name
        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
        else:
            output_dir = self.model_dir

        self.nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

@plac.annotations(
    data_name=("Location dataframe", "option", 'd'),
    model_name=("Where to find the model", "option", 'm'),
    out_name=("where to save the model", 'option', 'o'),
    reset_model=("Reset model found in model_loc", "flag", "r")
)
def main(   data_name='articles.pkl',
            model_name='spacy_clf',
            out_name='spacy_clf',
            reset_model=False):

    model = Model(model_name, reset_model)
    model.fit(data_name, n_epochs=1)
    model.save(out_name)




if __name__ == '__main__':
    plac.call(main)
