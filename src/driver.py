import formatter as ft
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.naive_bayes import MultinomialNB as MNB
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.pipeline import make_pipeline
from pprint import pprint
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity as cs

def cos_sim(tfidf, X, fox_df, hp_df, reu_df):
    tfidf.fit(X)
    fox = tfidf.transform(fox_df.content.values)
    hp = tfidf.transform(hp_df.content.values)
    reu = tfidf.transform(reu_df.content.values)

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


if __name__ == '__main__':

    fox_df, hp_df, reu_df = ft.loader_formatter()

    fox_df['bias'] = 1
    reu_df['bias'] = 0
    hp_df['bias'] = -1

    X = np.hstack( [fox_df.content.values, reu_df.content.values, hp_df.content.values] )
    y = np.hstack( [fox_df.bias.values, reu_df.bias.values, hp_df.bias.values])

    n = X.shape[0]
    inds = np.arange(n)
    np.random.shuffle(inds)
    X = X[inds]
    y = y[inds]

    X_t, X_e, y_t, y_e = train_test_split(X, y, test_size = 0.2)

    vectorizer = Tfidf(stop_words='english', norm='l2', max_df = 0.9)
    model = MNB()
    clf = make_pipeline(vectorizer, model)

    y_p = cross_val_predict( clf, X_t, y_t, cv=3, n_jobs=4 )
    pprint( classification_report(y_t, y_p) )

    cos_sim(vectorizer, X, fox_df, hp_df, reu_df )
