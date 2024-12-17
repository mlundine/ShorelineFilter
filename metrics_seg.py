import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from kneed import KneeLocator


def loss_curve(history):
    rgb_df = pd.read_csv(history)
    plt.rcParams["figure.figsize"] = (8,8)
    ##loss
    plt.subplot(4, 1, 1)
    plt.plot(range(1,len(rgb_df)+1), rgb_df['loss'], color='blue', label='Training')
    plt.plot(range(1,len(rgb_df)+1), rgb_df['val_loss'], color='red', label='Validation')
    idx = np.argmin(rgb_df['val_loss'])
    idx = idx+1
    plt.axvline(x=idx, color='k', label='Best')
    plt.xlim(1,len(rgb_df)+1) 
    plt.xlabel('Epoch')
    plt.ylabel('Loss (BCE)')
    plt.legend(loc='best', prop={'size': 6})
    #accuracy
    plt.subplot(4, 1, 2)
    plt.plot(range(1,len(rgb_df)+1), rgb_df['acc'], color='blue', label='Training')
    plt.plot(range(1,len(rgb_df)+1), rgb_df['val_acc'], color='red', label='Validation')
    plt.axvline(x=idx, color='k', label='Best')
    plt.xlim(1,len(rgb_df)+1)    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best',prop={'size': 6})
    #precision
    plt.subplot(4, 1, 3)
    plt.plot(range(1,len(rgb_df)+1), rgb_df['precision'], color='blue', label='Training')
    plt.plot(range(1,len(rgb_df)+1), rgb_df['val_precision'], color='red', label='Validation')
    plt.axvline(x=idx, color='k', label='Best')
    plt.xlim(1,len(rgb_df)+1) 
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend(loc='best',prop={'size': 6})
    #recall
    plt.subplot(4, 1, 4)
    plt.plot(range(1,len(rgb_df)+1), rgb_df['recall'], color='blue', label='Training')
    plt.plot(range(1,len(rgb_df)+1), rgb_df['val_recall'], color='red', label='Validation')
    plt.axvline(x=idx, color='k', label='Best')
    plt.xlim(1,len(rgb_df)+1) 
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend(loc='best', prop={'size': 6})
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'test_results', 'loss_curve.png'), dpi=500)
    plt.close('all')
    
def optimize_scores(test_scores_good_rgb_path,
                    test_scores_bad_rgb_path):
    plt.rcParams["figure.figsize"] = (10,5)
    ##rgb model
    test_scores_good_rgb = pd.read_csv(test_scores_good_rgb_path)
    test_scores_bad_rgb = pd.read_csv(test_scores_bad_rgb_path)
    thresholds = np.arange(0, 1.001, .001)
    good_filter_tps = [None]*len(thresholds)
    good_filter_fns = [None]*len(thresholds)
    bad_filter_tps = [None]*len(thresholds)
    bad_filter_fns = [None]*len(thresholds)
    i=0
    for thresh in thresholds:
        good_filter_tp = test_scores_good_rgb[test_scores_good_rgb['model_scores']>=thresh]
        good_filter_tps[i] = len(good_filter_tp)
        good_filter_fn = test_scores_good_rgb[test_scores_good_rgb['model_scores']<thresh]
        good_filter_fns[i] = len(good_filter_fn)
        bad_filter_tp = test_scores_bad_rgb[test_scores_bad_rgb['model_scores']<thresh]
        bad_filter_tps[i] = len(bad_filter_tp)
        bad_filter_fn = test_scores_bad_rgb[test_scores_bad_rgb['model_scores']>=thresh]
        bad_filter_fns[i] = len(bad_filter_fn)
        i=i+1
    good_filter_tps = np.array(good_filter_tps)
    good_filter_fns = np.array(good_filter_fns)
    bad_filter_tps = np.array(bad_filter_tps)
    bad_filter_fns = np.array(bad_filter_fns)  
    plt.subplot(1,2,1)
    plt.plot(thresholds, good_filter_tps, color='blue', label='Good')
    plt.plot(thresholds, bad_filter_tps, color='red', label='Bad')
    idx = np.argwhere(np.diff(np.sign(good_filter_tps - bad_filter_tps))).flatten()
    idx2 = np.argmin(np.diff(good_filter_tps)).flatten()
    search = np.arange(0, len(good_filter_tps))
    idx2 = KneeLocator(search,
                          list(good_filter_tps),
                          online=True,
                          curve='concave',
                          direction='decreasing',
                          interp_method='polynomial'
                          ).knee
    print(idx2)
    print(thresholds[idx])
    print(thresholds[idx2])
    lab = 'Optimum Threshold = '+str(np.round(thresholds[idx][0],3))
    lab2 = 'Knee Threshold = '+str(np.round(thresholds[idx2],3))
    plt.plot(thresholds[idx[0]], good_filter_tps[idx[0]], 'ro', color='k', label=lab)
    plt.plot(thresholds[idx2], good_filter_tps[idx2], 'ro', color='k', label=lab2)
    plt.xticks(np.arange(0,1.1,0.1))
    plt.xlabel('Threshold')
    plt.ylabel('True Positives')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(thresholds, good_filter_fns, color='blue', label='Good')
    plt.plot(thresholds, bad_filter_fns, color='red', label='Bad')
    idx = np.argwhere(np.diff(np.sign(good_filter_fns - bad_filter_fns))).flatten()
    print(thresholds[idx])
    lab = 'Optimum Threshold = '+str(np.round(thresholds[idx][0],3))
    plt.plot(thresholds[idx[0]], good_filter_fns[idx[0]], 'ro', color='k', label=lab)
    plt.xlabel('Threshold')
    plt.ylabel('False Negatives')
    plt.xticks(np.arange(0,1.1,0.1))
    plt.legend()
    plt.savefig(os.path.join(os.getcwd(), 'test_results', 'optimize_scores.png'), dpi=500)
    plt.close('all')

def test_scores_plot(test_scores_path_rgb_good, test_scores_path_rgb_bad):
    plt.rcParams["figure.figsize"] = (10,10)
    test_scores_rgb = pd.concat([pd.read_csv(test_scores_path_rgb_good), pd.read_csv(test_scores_path_rgb_bad)])
    plt.hist(test_scores_rgb['model_scores'], bins=100)
    plt.xticks(np.arange(0,1.1,0.1))
    plt.xlabel('Test Dataset Sigmoid Scores')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.savefig(os.path.join(os.getcwd(), 'test_results', 'test_scores.png'), dpi=500)
    plt.close('all')
    
def roc_curve(test_scores_good_rgb_path,
              test_scores_bad_rgb_path):
    plt.rcParams["figure.figsize"] = (8,8)
    ##rgb model
    test_scores_good_rgb = pd.read_csv(test_scores_good_rgb_path)
    test_scores_bad_rgb = pd.read_csv(test_scores_bad_rgb_path)
    thresholds = np.arange(0, 1.001, .001)
    tps = [None]*len(thresholds)
    fps = [None]*len(thresholds)
    fns = [None]*len(thresholds)
    tns = [None]*len(thresholds)
    sensitivities = [None]*len(thresholds)
    specificities = [None]*len(thresholds)
    
    i=0
    for thresh in thresholds:
        tp = len(test_scores_good_rgb[test_scores_good_rgb['model_scores']>=thresh])
        tps[i] = tp
        fp = len(test_scores_bad_rgb[test_scores_bad_rgb['model_scores']>=thresh])
        fps[i] = fp
        fn = len(test_scores_good_rgb[test_scores_good_rgb['model_scores']<thresh])
        fns[i] = fn
        tn = len(test_scores_bad_rgb[test_scores_bad_rgb['model_scores']<thresh])
        tns[i] = tn
        sensitivity = tp/(tp+fn)
        specificity = tn/(tn+fp)
        sensitivities[i] = sensitivity
        specificities[i] = specificity
        i=i+1

    tps = np.array(tps)
    fps = np.array(fps)
    fns = np.array(fns)
    tns = np.array(tns)
    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)
    
    P = len(test_scores_good_rgb)
    N = len(test_scores_bad_rgb)
    
    plt.plot(fps/N, tps/P, label='RGB Model')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'test_results', 'roc_curve.png'), dpi=500)
    plt.close('all')
                 
##Making plots
test_results = os.path.join(os.getcwd(), 'test_results')
history_path = os.path.join(os.getcwd(), 'models', 'history.csv')
test_scores_path_rgb_good = os.path.join(test_results, 'result_test_good_seg.csv')
test_scores_path_rgb_bad = os.path.join(test_results, 'result_test_bad_seg.csv')
loss_curve(history_path)
optimize_scores(test_scores_path_rgb_good,
                test_scores_path_rgb_bad)
test_scores_plot(test_scores_path_rgb_good, test_scores_path_rgb_bad)
roc_curve(test_scores_path_rgb_good,
          test_scores_path_rgb_bad)
