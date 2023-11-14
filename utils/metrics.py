import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import os

def compute_eer(fpr,tpr,thresholds):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    fnr = 1-tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((fpr[min_index], fnr[min_index]))
    return eer, thresholds[min_index]


def ComputeMetric(y_true, y_score, pos_label=1, isPlot=False, model_name='estimator', fig_path='.'):
    auc = metrics.roc_auc_score(y_true, y_score)
    
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=pos_label)
    eer, best_thresh = compute_eer(fpr, tpr, thresholds)
    
    if isPlot:
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc,
                                          estimator_name=model_name)
        display.plot()
        if not fig_path is None:
            plt.savefig(os.path.join(fig_path, model_name+'.png'))
        plt.show()

    return auc, eer, best_thresh