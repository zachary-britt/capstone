import baseline_model
import pandas as pd
import zutils
import eval_utils
from spacy_textcat import Spacecat
import plac
import ipdb
from pathlib import Path
from pprint import pprint
import os
DATA_PATH = os.environ['DATA_PATH']


def run_baseline_model():
    baseline_model.main()


def cross_val_execute(holdout_source, **cfg):
    source = holdout_source
    print('\n\nCROSS VALIDATION MODEL')
    print('Holdout set: {}\n'.format(source))
    train_name = 'cross_vals/{}/train.pkl'.format(source)
    test_name = 'cross_vals/{}/test.pkl'.format(source)
    model_out_name = out_name + source + '.2'
    cfg['out_name'] = model_out_name
    cfg['train_all'] = True
    cfg['test_all'] = False

    base_model_name = 'cross_val_models/'+source

    spacecat = Spacecat(base_model_name, **cfg)
    spacecat.fit(train_name, **cfg)
    spacecat.save()

    cfg['train_all'] = False
    cfg['test_all'] = True
    cfg['catnest'] = True
    # test_data = zutils.load_and_configure_data(test_name, **cfg)['test']
    test_data = pd.read_pickle(DATA_PATH+'cross_vals/{}/test.pkl'.format(source))
    #test_data = test_data.head(1000)
    test_scores = spacecat.evaluate_confusion(test_data)
    test_df = pd.read_pickle(DATA_PATH+'cross_vals/{}/test.pkl'.format(source))
    #test_df = test_df.head(1000)
    #ipdb.set_trace()
    test_df['left_pred'] = test_scores['left']
    test_df['right_pred'] = test_scores['right']
    print('\n{} predicted scores:\n'.format(source))
    pprint(test_df[['left_pred','right_pred']].describe())
    test_df.to_pickle(DATA_PATH+'cross_vals/{}/test_scores2.pkl'.format(source))




@plac.annotations(
    base_model_name=("Where to find the base model", "option", 'm', str),
    out_name=("where to save the model", 'option', 'o', str),
    resampling=('Type of resampling to use [over, under, choose_n, none]', 'option', 'rs', str),
    max_class_size=('max class size for choose N resampling', 'option', 'maxN', int),
    dropout=("Dropout rate to use", 'option', 'do', float),
    L2_penalty=("L2 regularization param", 'option', 'l2', float),
    minb=("Minimum Batch size", 'option', "minb", float),
    maxb=("Maximum Batch size", 'option', "maxb", float),
    float_bias=('Use float proportional bias', 'flag', 'fb', bool),
    epochs=("Training epochs", 'option', 'ep', int),
    quiet=('Dont print all over everything','flag','q', bool),
    reddit_ratio=('ratio of reddit comments to articles in mix','option','rr',float)
)
def main(   base_model_name='spacy-15.r4',
            out_name='cross_val_models/',
            resampling='under',
            max_class_size=10000,
            dropout=0.5,
            L2_penalty=1e-6,
            min_batch_size=4.,
            max_batch_size=64.,
            float_bias=False,
            epochs=1,
            quiet=False,
            reddit_ratio=5.0):

    '''Cross validation setup and execution'''

    # configuraiton setup
    if float_bias:
        label_type = 'catbias'
    else:
        label_type = 'cats'
    verbose = not quiet
    zipit=True
    cfg = dict(locals())

    # Create mixed cross validation training sets
    zutils.make_ultra_cross_val(reddit_ratio=reddit_ratio)

    out_dir = DATA_PATH + 'model_cache/'+ out_name
    out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir()

    sources = ['hp','mj','od','ai','reu','nyt','fox','bb','gp','cnn']
    for source in sources:
        cross_val_execute(source, **cfg)

if __name__ == '__main__':
    plac.call(main)









    #
