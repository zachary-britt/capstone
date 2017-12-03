import baseline_model
import zutils
import eval_utils
from spacy_textcat import Model as Spacecat
import plac
import ipdb

def run_baseline_model():
    baseline_model.main()


@plac.annotations(
    base_model_name=("Where to find the base model", "option", 'm', str),
    out_name=("where to save the model", 'option', 'o', str),
    resampling=('Type of resampling to use [over, under, choose_n, none]', 'option', 'rs', str),
    maxN=('max class size for choose N resampling', 'option', 'maxN', int),
    dropout=("Dropout rate to use", 'option', 'do', float),
    L2_penalty=("L2 regularization param", 'option', 'l2', float),
    min_batch_size=("Minimum Batch size", 'option', "minb", float),
    max_batch_size=("Maximum Batch size", 'option', "maxb", float),
    float_bias=('Use float proportional bias', 'flag', 'fb', bool),
    epochs=("Training epochs", 'option', 'ep', int),
    quiet=('Dont print all over everything','flag','q', bool),
    reddit_ratio=('ratio of reddit comments to articles in mix','option','rr',float),
    val_size=('train_set validation proportion', 'option','vs', float)
)
def main(   base_model_name='spacy-13.r5',
            out_name='cross-val-models/',
            resampling='under',
            maxN=10000,
            dropout=0.5,
            L2_penalty=1e-6,
            min_batch_size=4.,
            max_batch_size=64.,
            float_bias=False,
            epochs=1,
            quiet=False,
            reddit_ratio=5,
            val_size=0.05):

    #zutils.make_ultra_cross_val(reddit_ratio=reddit_ratio)


    if float_bias:
        label_type = 'catbias'
    else:
        label_type = 'cats'

    cfg = {     'resampling':resampling, 'max_class_size': maxN,'dropout':dropout,
                'minb':min_batch_size,'maxb':max_batch_size, 'label_type':label_type,
                'epochs':epochs,'verbose': not quiet,'L2_penalty':L2_penalty, "zipit":True,
                }


    sources = ['hp','mj','od','ai','reu','nyt','fox','bb','gp','cnn']
    for source in sources:
        train_name = 'cross_vals/{}/train.pkl'.format(source)
        test_name = 'cross_vals/{}/test.pkl'.format(source)
        model_out_name = out_name + source
        cfg['out_name'] = model_out_name

        spacecat = Spacecat(base_model_name, **cfg)
        spacecat.fit(train_name, **cfg)
        spacecat.save(out_name)

        test_data = zutils.load_and_configure_data(test_name, **spacecat.cfg)['test']
        spacecat.evaluate_confusion(data)

if __name__ == '__main__':
    plac.call(main)









    #
