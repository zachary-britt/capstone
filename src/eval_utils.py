import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pickle
import ipdb
from pathlib import Path

def make_roc(y_true, y_pred, label, N=None, title=None, action=None, file_path=None):
    '''
    Produces ROC curve. Call multiple times if needed. Requires user to plt.show()

    '''

    inds = np.argsort(y_pred)
    y_pred = y_pred[inds]
    y_true = y_true[inds]

    threshes = np.hstack([y_pred.min()-1e-6, *y_pred, 1 ])[::-1]
    Ms = [build_confusion(y_true, y_pred,t) for t in threshes]
    Ms = pd.DataFrame(Ms)
    n = y_true.shape[0]
    t = Ms.tp[0] + Ms.fn[0]
    f = Ms.fp[0] + Ms.tn[0]
    t = max(t,1e-8)
    f = max(f,1e-8)
    tprs = Ms.tp.values / t
    fprs = Ms.fp.values / f

    #calc roc area (fun tiny integral calc)
    dx= fprs[1:] - fprs[:-1]

    area = ((tprs[1:]+tprs[:-1])*0.5 * dx).sum()
    area_str = '{}: Area = {:.3}'.format(label, area)

    # return {'x':fprs, 'y': tprs, 'label':area_str, 'N':N}

    if label=='left':
        c = 'b'
    elif label=='right':
        c = 'r'
    else:
        c = 'p'

    plt.plot(fprs, tprs, label=area_str, c=c )
    plt.plot([0,1], [0,1], '--', c='g' )
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    if title:
        plt.suptitle(title)
    elif N:
        plt.suptitle('ROC Curve after {} samples trained on'.format(N))
    else:
        plt.suptitle('ROC Curve')
    plt.legend()

    if action=='show':
        plt.show()
    elif file_path:
        dir_ = Path(file_path)
        if not dir_.exists():
            dir_.mkdir()

        index = 0
        path = Path("{}/{}.png".format(file_path, index))
        while path.exists():
            index += 1
            path = Path("{}/{}.png".format(file_path, index))

        plt.savefig(str(path), bbox_inches='tight')
        plt.clf()



def build_confusion(y_true, y_pred, thresh):
    '''
    Creates a confusion matrix (as a dictionary)
    '''
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.min() < 0:
        y_true = np.where(y_true > 0, 1, 0)

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
    '''
    Calculates precision, recall and F1 score
    '''
    tp = M.get('tp',1e-6)
    fp = M.get('fp',1e-6)
    tn = M.get('tn',1e-6)
    fn = M.get('fn',1e-6)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f_score


def print_confusion_report(M, name=None):
    '''
    Nicely prints a the PRF report
    '''
    if name:
        print('\nScores for: {}'.format(name))

    P, R, F = calc_PRF(M)

    print('{:^5}\t{:^5}\t{:^5}'.format('P', 'R', 'F'))
    print('{0:.3f}\t{1:.3f}\t{2:.3f}'.format(P, R, F))


def build_label_profile(y_true, y_preds, label, Ns):
    rocs = []
    for i, y_pred in enumerate(y_preds):
       rocs.append( make_roc(y_true, y_pred, label, Ns[i]) )
    # make_roc(y_true, y_preds[0], label, Ns[0])
    #build_animation(rocs)

def score_average(texts, models):
    scores_list = [model.score_texts(texts) for model in models]
    scores_dfs = [pd.DataFrame(scores) for scores in scores_list]
    scores_df = scores_dfs[0]
    for next_score_df in scores_dfs[1:]:
        scores_df['left'] = scores_df['left'] + next_score_df['left']
        scores_df['right']=scores_df['right']+next_score_df['right']
    scores_df['aleft'] = scores_df['left']/9
    scores_df['aright']=scores_df['right']/9
    return scores_df


# def build_animation(rocs):
#     fig, ax = plt.subplots()
#     ln, = ax.plot([], [], 'b', animated=True)
#     #ipdb.set_trace()
#
#     def init():
#         ax.set_xlim(0, 1)
#         ax.set_ylim(0, 1)
#         return ln,
#
#     def update(roc):
#         x = roc['x']
#         y = roc['y']
#         label = roc['label']
#         # ln.set_data(x, y)
#         # ln.set_label(label)
#         #
#         #fig.suptitle(roc['N'])
#         ax.clear()
#         ax.plot(x,y,c='b',label=label)
#         ax.figure.suptitle('ROC after N={} samples seen'.format(roc['N']))
#
#         # plt.legend(bbox_to_anchor=(0.9, 1.0, 0.0, .10), loc=1,
#         #     mode="expand", borderaxespad=0.)
#
#         plt.legend()
#         return ax,
#
#     ani = FuncAnimation(fig, update, frames=iter(rocs), blit=True, init_func=init)
#     plt.show()



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

    make_roc(y_true_left, l_scores_arr[-1], 'left', Ns[-1])
    make_roc(y_true_right, r_scores_arr[-1], 'right', Ns[-1])
    plt.show()


def scale_back_reu(scores):
    mid_ = scores.loc['reu']
    shift = -mid_
    scores = scores + shift

    min_ = scores.min()
    max_ = scores.max()

    for source,score in zip(scores.index, scores):
        if score<0:
            scores.loc[source] = score * 1/(-min_)
        else:
            scores.loc[source] = score * 1/max_

    return scores


if __name__ == '__main__':

    # model_names = ['spacy-3.aaara', 'spacy-4.raaaa', 'spacy-5.a10', 'spacy-6.aaaa',
    #                     'spacy-7.aaaa', 'spacy-8.ara']
    model_name = 'spacy-8.ara'
    build_profile(model_name)




    #
