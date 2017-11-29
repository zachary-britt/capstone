import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import ipdb


def make_roc(y_true, y_pred, label, N):
    ipdb.set_trace()
    inds = np.argsort(y_pred)
    y_pred = y_pred[inds]
    y_true = y_true[inds]

    threshes = np.hstack([0, *y_pred, 1 ])[::-1]
    Ms = [build_confusion(y_true, y_pred,t) for t in threshes]
    Ms = pd.DataFrame(Ms)
    n = y_true.shape[0]
    t = max(y_true.sum(),1e-8)
    f = max(n-t,1e-8)
    tprs = Ms.tp.values/t
    fprs = Ms.fp.values/f

    #calc roc area (fun tiny integral calc)
    dx= fprs[1:] - fprs[:-1]
    area = (tprs[1:] * dx).sum()
    area_str = '{}: Area = {}'.format(label, area)

    plt.plot(fprs, tprs, label=area_str)
    plt.plot(threshes, threshes, '--', c='r' )
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()




def build_confusion(y_true, y_pred, thresh):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tpi = np.where( y_true & (y_pred >= thresh), 1, 0)
    fpi = np.where( ~y_true & (y_pred >= thresh), 1, 0)
    tni = np.where( ~y_true & (y_pred < thresh), 1, 0)
    fni = np.where( y_true & (y_pred < thresh), 1, 0)

    tp = tpi.sum()
    fp = fpi.sum()
    tn = tni.sum()
    fn = fni.sum()
    N = len(y_true)

    M = {'tp':tp, 'fp':fp, 'tn':tn, 'fn': fn}
    return M


def calc_PRF(M):
    tp = M.get('tp',1e-8)
    fp = M.get('fp',1e-8)
    tn = M.get('tn',1e-8)
    fn = M.get('fn',1e-8)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f_score


def print_confusion_report(M, name=None):

    if name:
        print('\nScores for: {}'.format(name))

    P, R, F = calc_PRF(M)

    print('{:^5}\t{:^5}\t{:^5}'.format('P', 'R', 'F'))
    print('{0:.3f}\t{1:.3f}\t{2:.3f}'.format(P, R, F))


def build_label_profile(y_true, y_preds, label, Ns):

    rocs = []
    #for i, y_pred in enumerate(y_preds):
    #    rocs.append( make_roc(y_true, y_pred, label, Ns[0]) )
    make_roc(y_true, y_preds[0], label, Ns[0])



def build_profile(model_name):
    y_true_left = np.hstack([np.ones(250), np.zeros(250)]).astype(bool)
    y_true_right = y_true_left[::-1]

    path = '../data/model_cache/{}/cont_val.pkl'.format(model_name)
    with open(path,'rb') as f:
        scores_list, epoch_seq, N = pickle.load(f)

    r_scores_list = [[score['right'] for score in scores[1]] for scores in scores_list]
    l_scores_list = [[score['left'] for score in scores[1]] for scores in scores_list]
    Ns = [scores[0] for scores in scores_list]

    l_scores_arr = np.array(l_scores_list)
    r_scores_arr = np.array(r_scores_list)

    build_label_profile(y_true_left, l_scores_arr, 'left', Ns)

if __name__ == '__main__':

    model_names = ['spacy-3.aaara', 'spacy-4.raaaa', 'spacy-5.a10', 'spacy-6.aaaa',
                        'spacy-7.aaaa', 'spacy-8.arar']

    build_profile(model_names[0])






    #
