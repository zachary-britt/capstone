from __future__ import unicode_literals, print_function

import spacy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from spacy.util import minibatch, compounding
from pathlib import Path
import ipdb

def load_text():
    df = pd.read_pickle('../data/formatted_arts.pkl')

    X = df.content.values
    y = df.bias.values

    # left_inds = np.argwhere( y == 'left' ).ravel()
    # right_inds = np.argwhere( y == 'right' ).ravel()
    # center_inds = np.argwhere( y == 'center' ).ravel()
    #
    #
    # N_l = left_inds.shape[0]; N_r = right_inds.shape[0]
    # N = N_l + N_r
    #
    # l_frac = N_l / N; r_frac = N_r / N;
    # l_weight = 1 / l_frac; r_weight = 1 / r_frac;
    #
    # print('Suport:\n \t left: \t {} \n \t right:\t {}'.format(N_l, N_r))
    # print('Weight:\n \t left: \t {} \n \t right:\t {}'.format(l_weight, r_weight))
    #
    # lr_inds = np.hstack([left_inds, right_inds])
    # np.random.shuffle(lr_inds)
    #
    # X = X[lr_inds]
    # y = y[lr_inds]

    y = [{'left': bias=='left', 'center': bias=='center', 'right': bias=='right'} for bias in y]

    # y = np.array([{'left': bias=='left'} for bias in y])

    return train_test_split(X,y, test_size=0.2)



''' borrowed '''
def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 1e-8  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 1e-8  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
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




if __name__ == '__main__':
    nlp = spacy.load('en_core_web_lg')
    textcat = nlp.create_pipe('textcat')
    nlp.add_pipe(textcat, last=True)
    textcat.add_label('left')
    textcat.add_label('center')
    textcat.add_label('right')

    #ipdb.set_trace()

    train_texts, test_texts, train_cats, test_cats = load_text()

    test_texts, val_texts, test_cats, val_cats = train_test_split(test_texts, test_cats, test_size=0.2)

    n_samples = train_texts.shape[0]

    n_iter = 2

    '''Borrowed'''
    train_data = list(zip(train_texts, [{'cats': cats} for cats in train_cats]))

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training(n_workers = -1)
        print("Training the model...")
        print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))
        seen = 0
        for i in range(n_iter):
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4., 64., 1.01))
            #batches = minibatch(train_data, size=124)
            for j,batch in enumerate(batches):
                texts, annotations = zip(*batch)
                seen += len(texts)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
                           losses=losses)
                if seen > 5000:
                    seen = 0
                    with textcat.model.use_params(optimizer.averages):
                        # evaluate on the dev data split off in load_data()
                        scores = evaluate(nlp.tokenizer, textcat, val_texts, val_cats)
                    print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'  # print a simple table
                          .format(losses['textcat'], scores['textcat_p'],
                                  scores['textcat_r'], scores['textcat_f']))
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, test_texts, test_cats)
            print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'  # print a simple table
                  .format(losses['textcat'], scores['textcat_p'],
                          scores['textcat_r'], scores['textcat_f']))


    # test the trained model
    test_text = "Trump is keeping all of his campaign promises."
    doc = nlp(test_text)
    print(test_text, doc.cats)

    test_text = "Trump looks to be the worst president in history."
    doc = nlp(test_text)
    print(test_text, doc.cats)

    output_dir = '../data/spacy_clf/'

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        print(test_text, doc2.cats)







#
