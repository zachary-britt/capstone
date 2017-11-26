from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_predict
from pprint import pprint
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity as cs
import numpy as np
import plac
import os
DATA_PATH = os.environ['DATA_PATH']

def make_vectorizer():
    vectorizer = Tfidf(stop_words='english', norm='l2', max_df=0.7,
                        max_features=12000, sublinear_tf=True)
    return vectorizer

def tfidf_NB_baseline(X, y):

    vectorizer = make_vectorizer()
    model = MNB()
    clf = make_pipeline(vectorizer, model)

    y_p = cross_val_predict( clf, X, y, cv=3, n_jobs=4 )
    pprint( classification_report(y, y_p) )


def cos_sim(X, y):

    vectorizer = make_vectorizer()

    fox_inds = np.argwhere(y == 'right').ravel()
    hp_inds = np.argwhere(y == 'left').ravel()
    reu_inds = np.argwhere(y == 'center').ravel()

    X_fox = X[fox_inds]
    X_hp = X[hp_inds]
    X_reu = X[reu_inds]

    vectorizer.fit(X)
    fox = vectorizer.transform(X_fox)
    hp = vectorizer.transform(X_hp)
    reu = vectorizer.transform(X_reu)

    fox_n = fox.shape[0]
    hp_n = hp.shape[0]
    reu_n = reu.shape[0]

    fox_self_sim = cs(fox)
    hp_self_sim = cs(hp)
    reu_self_sim = cs(reu)

    fox_self_sim_score = (fox_self_sim.sum() - fox_n) / (fox_n * (fox_n -1))
    hp_self_sim_score = (hp_self_sim.sum() - hp_n) / (hp_n * (hp_n -1))
    reu_self_sim_score = (reu_self_sim.sum() - reu_n) / (reu_n * (reu_n -1))

    print('fox self: {}'.format(fox_self_sim_score))
    print('hp self: {}'.format(hp_self_sim_score))
    print('reu self: {}'.format(reu_self_sim_score))

    fox_hp_sim = cs(fox, hp)
    fox_reu_sim = cs(fox, reu)
    hp_reu_sim = cs(hp, reu)

    fox_hp_sim_score = fox_hp_sim.sum() / (fox_n * hp_n)
    fox_reu_sim_score = fox_reu_sim.sum() / (fox_n * reu_n)
    hp_reu_sim_score = hp_reu_sim.sum() / (reu_n * hp_n)

    print('fox and hp : {}'.format(fox_hp_sim_score))
    print('fox and reu : {}'.format(fox_reu_sim_score))
    print('hp and reu : {}'.format(hp_reu_sim_score))


def main(   train_data_loc=DATA_PATH+'articles.pkl',
            test_data_loc=DATA_PATH+'holdout.pkl'):

    X = df.content.values
    d = df.date.values
    y = df.bias.values
    

if __name__ == '__main__':
    plac.call(main)
