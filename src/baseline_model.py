from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_predict
from pprint import pprint
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity as cs
import numpy as np
import plac
import ipdb
import os
DATA_PATH = os.environ['DATA_PATH']

import zutils


class BaselineCLF:
    def __init__(self, **cfg):
        self.labels = cfg.get('labels',['left','right'])
        self.clf = {label: self._make_tfidf_NB_clf(**cfg) for label in self.labels}

    def _make_tfidf_NB_clf(self, **cfg):
        max_f = cfg.get('max_features',1200)
        max_df= cfg.get('max_df',0.7)
        sublin = cfg.get('sublin',True)
        vectorizer = Tfidf(stop_words='english', norm='l2', max_df=max_df,
                            max_features=max_f, sublinear_tf=sublin)
        model = MNB()
        clf = Pipeline( steps=[('v',vectorizer), ('nb',model) ])
        return clf

    # def _data_cfg(self, data):
    #     X,y = zip(*data)
    #     X = np.array(X)
    #     y = np.vstack(y)
    #     return (X,y)

    def _baseline_data_load_and_cfg(self, **cfg):
        data_cfg = {
            'label_type':   'one_hot',
            'train_all':    True,
            'zipit':        True,
            'resampling':   'over'
        }
        cfg.update(data_cfg)

        if cfg.get('no_center'):    art_name = 'articles_no_center.pkl'
        else:                       art_name = 'articles.pkl'

        training_data = zutils.load_and_configure_data(art_name, **cfg)['train']
        class_weights = np.ones(len(training_data))

        if cfg.get('use_reddit'):
            cfg['resampling']='choose_n'
            cfg['max_class_size']=10000

            if cfg.get('no_center'):    red_name = 'reddit_no_center.pkl'
            else:                       red_name = 'reddit.pkl'

            reddit_data = zutils.load_and_configure_data(red_name, **cfg)['train']
            training_data.extend(reddit_data)
            reddit_weight = cfg.get('reddit_weight',0.2)
            class_weights = np.hstack([class_weights, reddit_weight*np.ones(len(reddit_data))])
        # training_data = self._data_cfg(training_data)

        peek = cfg.get('peek_cfg')
        testing_data = zutils.load_and_configure_data('holdout.pkl', peek=peek , **cfg)['test']
        # testing_data = self._data_cfg(testing_data)

        data = {'train':training_data, 'test':testing_data, 'weights':class_weights}

        return data


    def train(self, data=None, **cfg):
        if data:
            X_t, y_t = zip(*data['train'])
            X_e, y_e = zip(*data['test'])
            W = data.get('weights',None)
        else:
            data = self._baseline_data_load_and_cfg(**cfg)
            X_t, y_t = zip(*data['train'])
            X_e, y_e = zip(*data['test'])
            W = data.get('weights',None)

        y_t = np.vstack(y_t)
        y_e = np.vstack(y_e)


        for i,label in enumerate(self.labels):
            print("\n\n{} VS ALL".format(label.upper()))

            y_t_1d = y_t[:,i]
            y_e_1d = y_e[:,i]

            self.clf[label].fit(X_t, y_t_1d, **{'nb__sample_weight':W})

            y_train_preds = self.clf[label].predict(X_t)
            y_test_pred = self.clf[label].predict(X_e)

            train_report= classification_report(y_t_1d, y_train_preds, target_names=['not {}'.format(label),label])
            test_report = classification_report(y_e_1d, y_test_pred, target_names=['not {}'.format(label),label])

            print('\nTRAINING REPORT\n')
            pprint(train_report)

            #ipdb.set_trace()

            print('\nTESTING REPORT\n')
            pprint(test_report)




def main():
    cfg = { 'use_reddit':False,
            'no_center':True,
            'reddit_weight':0.2,
            'peek_cfg':True
            }
    clf = BaselineCLF(**cfg)
    clf.train(**cfg)


if __name__ == '__main__':
    plac.call(main)
