import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def make_roc_plot(y_true, y_pred, label):
    threshes = np.linspace(0, 1)
    Ms = [build_confusion(y_true, y_pred,t) for t in threshes]

    PRFs = [calc_PRF(M) for M in Ms]
    P, R, F = zip(*PRFs)
    P, R, F = np.array(P), np.array(R), np.array(F)

    plt.plot(R, P, label = label)
    plt.plot(1-threshes, threshes, '--', c='r' )
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



if __name__ == '__main__':
    main()
