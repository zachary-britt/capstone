import spacy
import pandas as pd
import numpy as np
from spacy.util import minibatch, compounding
from pathlib import Path
import plac
import os
DATA_PATH = os.environ['DATA_PATH']
from pprint import pprint
import ipdb
from sklearn.model_selection import train_test_split



class Model:

    def __init__(self, model_dir, overwrite_model=False):
        self.model_dir = Path(model_dir)
        self.labels = ['left','right']
        self.open_nlp_with_text_cat_(overwrite_model)


    def open_nlp_with_text_cat_(self, overwrite_model):
        if not self.model_dir.exists():
            self.model_dir.mkdir()
            overwrite_model = True
        if overwrite_model:
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


    def load_and_configure_training_data(self, data_loc):
        df = pd.read_pickle(data_loc)

        X = df.content.values
        # y = df.bias.values
        y = df.orient.values

        print('class support:')
        pprint(df.orient.value_counts())

        #shuffling:
        inds = np.arange(X.shape[0])
        np.random.seed(1234)
        np.random.shuffle(inds)
        X, y = X[inds], y[inds]
        y = [{  'left': slant=='left',
                'right':slant=='right'
                } for slant in y]

        # t:train, e:evaluate, v:validate
        X_t, X_e, y_t, y_e = train_test_split(X, y, test_size=0.2)
        X_e, X_v, y_e, y_v = train_test_split(X_e, y_e, test_size=0.05)

        train_data = list(zip(X_t, [{'cats': cats} for cats in y_t]))
        valid_data = list(zip(X_v, [{'cats': cats} for cats in y_v]))
        evali_data = list(zip(X_e, [{'cats': cats} for cats in y_e]))

        return train_data, valid_data, evali_data

    def fit(self, data_loc, n_iter):



        train_data, valid_data, evali_data = self.load_and_configure_training_data(data_loc)

        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != 'textcat']
        with self.nlp.disable_pipes(*other_pipes):  # only train textcat
            optimizer = self.nlp.begin_training(n_workers = -1)
            print("Training the model...")
            print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))
            seen = 0
            for i in range(n_iter):
                losses = {}
                # batch up the examples using spaCy's minibatch
                batches = minibatch(train_data, size=compounding(4., 64., 1.01))
                for j,batch in enumerate(batches):
                    texts, annotations = zip(*batch)

                    self.nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
                            losses=losses)

                    ''' in process validation so there's something to watch '''
                    seen += len(texts)
                    if seen > 5000:
                        seen = 0
                        with self.textcat.model.use_params(optimizer.averages):
                            # validate
                            scores = self.evaluate(valid_data)
                            print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'
                                .format(losses['textcat'], scores['textcat_p'],
                                scores['textcat_r'], scores['textcat_f']))

                # end of epoch report
                with self.textcat.model.use_params(optimizer.averages):
                    # evaluate on the evaluation data
                    scores = self.evaluate(evali_data)
                    print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'
                        .format(losses['textcat'], scores['textcat_p'],
                        scores['textcat_r'], scores['textcat_f']))


    def evaluate(self, annotated_data):
        #ipdb.set_trace()
        texts, cat_dicts = zip(*annotated_data)
        docs = (self.nlp.tokenizer(text) for text in texts)
        tp = 1e-8  # True positives
        fp = 1e-8  # False positives
        fn = 1e-8  # False negatives
        tn = 1e-8  # True negatives
        for i, doc in enumerate(self.textcat.pipe(docs)):
            gold = cat_dicts[i]['cats']
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
        return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score}


    def save(self, output_dir=None):
        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
        else:
            output_dir = self.model_dir

        self.nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

@plac.annotations(
    data_loc=("Location dataframe"),
    model_loc=("Where to store model"),
    overwrite_model=("Overwrite model found in model_loc", "flag", "w")
)
def main(   data_loc=DATA_PATH+'formatted_arts.pkl',
            model_loc=DATA_PATH+'spacy_clf',
            overwrite_model=False):

    model = Model(model_loc, overwrite_model)
    model.fit(data_loc, n_iter=1)
    model.save()




if __name__ == '__main__':
    plac.call(main)
