import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from kneed import KneeLocator

def loss_curve(rgb_history, gray_history):
    rgb_df = pd.read_csv(rgb_history)
    gray_df = pd.read_csv(gray_history)
    plt.rcParams["figure.figsize"] = (12,15)
    ##loss
    plt.subplot(4, 2, 1)
    plt.plot(range(1,len(rgb_df)+1), rgb_df['loss'], color='blue', label='RGB Training')
    plt.plot(range(1,len(rgb_df)+1), rgb_df['val_loss'], color='red', label='RGB Validation')
    idx = np.argmin(rgb_df['val_loss'])
    idx = idx+1
    idx2 = np.argmin(gray_df['val_loss'])
    idx2 = idx2+1
    plt.axvline(x=idx, color='k', label='Best')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (BCE)')
    plt.legend(loc='best', prop={'size': 6})
    plt.subplot(4, 2, 2)
    plt.plot(range(1,len(gray_df)+1), gray_df['loss'], color='blue', label='Grayscale Training')
    plt.plot(range(1,len(gray_df)+1), gray_df['val_loss'], color='red', label='Grayscale Validation')
    plt.xticks(range(1,len(gray_df)+1))
    plt.axvline(x=idx2, color='k', label='Best')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (BCE)')
    plt.legend(loc='best',prop={'size': 6})
    #accuracy
    plt.subplot(4, 2, 3)
    plt.plot(range(1,len(rgb_df)+1), rgb_df['acc'], color='blue', label='RGB Training')
    plt.plot(range(1,len(rgb_df)+1), rgb_df['val_acc'], color='red', label='RGB Validation')
    plt.axvline(x=idx, color='k', label='Best')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best',prop={'size': 6})
    plt.subplot(4, 2, 4)
    plt.plot(range(1,len(gray_df)+1), gray_df['acc'], color='blue', label='Grayscale Training')
    plt.plot(range(1,len(gray_df)+1), gray_df['val_acc'], color='red', label='Grayscale Validation')
    plt.xticks(range(1,len(gray_df)+1))
    plt.axvline(x=idx2, color='k', label='Best')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='best',prop={'size': 6})
    #precision
    plt.subplot(4, 2, 5)
    plt.plot(range(1,len(rgb_df)+1), rgb_df['precision'], color='blue', label='RGB Training')
    plt.plot(range(1,len(rgb_df)+1), rgb_df['val_precision'], color='red', label='RGB Validation')
    plt.axvline(x=idx, color='k', label='Best')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend(loc='best',prop={'size': 6})
    plt.subplot(4, 2, 6)
    plt.plot(range(1,len(gray_df)+1), gray_df['precision'], color='blue', label='Grayscale Training')
    plt.plot(range(1,len(gray_df)+1), gray_df['val_precision'], color='red', label='Grayscale Validation')
    plt.xticks(range(1,len(gray_df)+1))
    plt.axvline(x=idx2, color='k', label='Best')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend(loc='best', prop={'size': 6})
    #recall
    plt.subplot(4, 2, 7)
    plt.plot(range(1,len(rgb_df)+1), rgb_df['recall'], color='blue', label='RGB Training')
    plt.plot(range(1,len(rgb_df)+1), rgb_df['val_recall'], color='red', label='RGB Validation')
    plt.axvline(x=idx, color='k', label='Best')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend(loc='best',prop={'size': 6})
    plt.subplot(4, 2, 8)
    plt.plot(range(1,len(gray_df)+1), gray_df['recall'], color='blue', label='Grayscale Training')
    plt.plot(range(1,len(gray_df)+1), gray_df['val_recall'], color='red', label='Grayscale Validation')
    plt.xticks(range(1,len(gray_df)+1))
    plt.axvline(x=idx2, color='k', label='Best')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend(loc='best', prop={'size': 6})
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'test_results', 'loss_curve.png'), dpi=500)
    plt.close('all')

def test_scores_plot(test_scores_path_rgb_good, 
                     test_scores_path_rgb_bad,
                     test_scores_path_gray_good,
                     test_scores_path_gray_bad):
    plt.rcParams["figure.figsize"] = (16,8)
    plt.subplot(1,2,1)
    plt.title('a) RGB Model')
    test_scores_rgb = pd.concat([pd.read_csv(test_scores_path_rgb_good), pd.read_csv(test_scores_path_rgb_bad)])
    plt.hist(test_scores_rgb['model_scores'], bins=100)
    plt.xticks(np.arange(0,1.1,0.1))
    plt.xlabel('Test Dataset Sigmoid Scores')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.subplot(1,2,2)
    plt.title('b) Grayscale Model')
    test_scores_gray = pd.concat([pd.read_csv(test_scores_path_gray_good), pd.read_csv(test_scores_path_gray_bad)])
    plt.hist(test_scores_gray['model_scores'], bins=100)
    plt.xticks(np.arange(0,1.1,0.1))
    plt.xlabel('Test Dataset Sigmoid Scores')
    plt.ylabel('Count')
    plt.yscale('log')   
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'test_results', 'test_scores.png'), dpi=500)
    plt.close('all')
    
def optimize_scores(test_scores_good_rgb_path,
                    test_scores_bad_rgb_path,
                    test_scores_good_gray_path,
                    test_scores_bad_gray_path):
    plt.rcParams["figure.figsize"] = (10,10)
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
    val=45
    good_filter_tps = np.array(good_filter_tps)
    good_filter_fns = np.array(good_filter_fns)
    bad_filter_tps = np.array(bad_filter_tps)
    bad_filter_fns = np.array(bad_filter_fns)  
    plt.subplot(2,2,1)
    plt.title('a) RGB Model')
    plt.plot(thresholds, good_filter_tps, color='blue', label='Suitable')
    plt.plot(thresholds, bad_filter_tps, color='red', label='Unsuitable')

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
    plt.subplot(2,2,2)
    plt.title('b) RGB Model')
    plt.plot(thresholds, good_filter_fns, color='blue', label='Suitable')
    plt.plot(thresholds, bad_filter_fns, color='red', label='Unsuitable')
    idx = np.argwhere(np.diff(np.sign(good_filter_fns - bad_filter_fns))).flatten()
    print(thresholds[idx])
    lab = 'Optimum Threshold = '+str(np.round(thresholds[idx][0],3))
    plt.plot(thresholds[idx], good_filter_fns[idx], 'ro', color='k', label=lab)
    plt.xlabel('Threshold')
    plt.ylabel('False Negatives')
    plt.xticks(np.arange(0,1.1,0.1))
    plt.legend()

    ##gray model
    test_scores_good_gray = pd.read_csv(test_scores_good_gray_path)
    test_scores_bad_gray = pd.read_csv(test_scores_bad_gray_path)
    good_filter_tps = [None]*len(thresholds)
    good_filter_fns = [None]*len(thresholds)
    bad_filter_tps = [None]*len(thresholds)
    bad_filter_fns = [None]*len(thresholds)
    i=0
    for thresh in thresholds:
        good_filter_tp = test_scores_good_gray[test_scores_good_gray['model_scores']>=thresh]
        good_filter_tps[i] = len(good_filter_tp)
        good_filter_fn = test_scores_good_gray[test_scores_good_gray['model_scores']<thresh]
        good_filter_fns[i] = len(good_filter_fn)
        bad_filter_tp = test_scores_bad_gray[test_scores_bad_gray['model_scores']<thresh]
        bad_filter_tps[i] = len(bad_filter_tp)
        bad_filter_fn = test_scores_bad_gray[test_scores_bad_gray['model_scores']>=thresh]
        bad_filter_fns[i] = len(bad_filter_fn)
        i=i+1
    good_filter_tps = np.array(good_filter_tps)
    good_filter_fns = np.array(good_filter_fns)
    bad_filter_tps = np.array(bad_filter_tps)
    bad_filter_fns = np.array(bad_filter_fns)  
    plt.subplot(2,2,3)
    plt.title('c) Grayscale Model')
    plt.plot(thresholds, good_filter_tps, color='blue', label='Suitable')
    plt.plot(thresholds, bad_filter_tps, color='red', label='Unsuitable')
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
    plt.subplot(2,2,4)
    plt.title('d) Grayscale Model')
    plt.plot(thresholds, good_filter_fns, color='blue', label='Suitable')
    plt.plot(thresholds, bad_filter_fns, color='red', label='Unsuitable')
    idx = np.argwhere(np.diff(np.sign(good_filter_fns - bad_filter_fns))).flatten()
    print(thresholds[idx])
    lab = 'Optimum Threshold = '+str(np.round(thresholds[idx][0],3))
    plt.plot(thresholds[idx], good_filter_fns[idx], 'ro', color='k', label=lab)
    plt.xlabel('Threshold')
    plt.ylabel('False Negatives')
    plt.xticks(np.arange(0,1.1,0.1))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'test_results', 'optimize_scores.png'), dpi=500)
    plt.close('all')
    

def get_random_images(result_test_path, print_string, good_or_bad):
    print(print_string)
    df = pd.read_csv(result_test_path)
    if good_or_bad == 'good':
        df = df[df['model_scores']>0.95]
    elif good_or_bad == 'bad':
        df = df[df['model_scores']<0.05]
    df.reset_index()
    idxes = [None]*10
    for i in range(10):
        idx = np.random.randint(0, len(df))
        while idx in idxes:
            idx = np.random.randind(0, len(df))
        print(df['im_paths'].iloc[idx])
        print(df['model_scores'].iloc[idx])
        
def get_random_edge_images(result_test_path, print_string, good_or_bad):
    print(print_string)
    df = pd.read_csv(result_test_path)
    df = df[df['model_scores']>0.25]
    df = df[df['model_scores']<0.45]
    df.reset_index()
    idxes = [None]*10
    for i in range(10):
        idx = np.random.randint(0, len(df))
        while idx in idxes:
            idx = np.random.randind(0, len(df))
        print(df['im_paths'].iloc[idx])
        print(df['model_scores'].iloc[idx])
        
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

def roc_curve(test_scores_good_rgb_path,
              test_scores_bad_rgb_path,
              test_scores_good_gray_path,
              test_scores_bad_gray_path):
    plt.rcParams["figure.figsize"] = (10,10)
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
    
    ##gray model
    test_scores_good_gray = pd.read_csv(test_scores_good_gray_path)
    test_scores_bad_gray = pd.read_csv(test_scores_bad_gray_path)
    tps = [None]*len(thresholds)
    fns = [None]*len(thresholds)
    fns = [None]*len(thresholds)
    tns = [None]*len(thresholds)
    sensitivities = [None]*len(thresholds)
    specificities = [None]*len(thresholds)
    
    i=0
    for thresh in thresholds:
        tp = len(test_scores_good_gray[test_scores_good_gray['model_scores']>=thresh])
        tps[i] = tp
        fp = len(test_scores_bad_gray[test_scores_bad_gray['model_scores']>=thresh])
        fps[i] = fp
        fn = len(test_scores_good_gray[test_scores_good_gray['model_scores']<thresh])
        fns[i] = fn
        tn = len(test_scores_bad_gray[test_scores_bad_gray['model_scores']<thresh])
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

    P = len(test_scores_good_gray)
    N = len(test_scores_bad_gray)
    plt.plot(fps/N, tps/P, label='Grayscale Model', color='k')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'test_results', 'roc_curve.png'), dpi=500)
    plt.close('all')

##results folder
result_folder = os.path.join(os.getcwd(), 'test_results')

##grayscale files
history_gray = os.path.join(os.getcwd(), 'models', 'gray', 'history.csv')
result_test_good_gray = os.path.join(result_folder, 'result_test_good_gray.csv')
result_test_bad_gray = os.path.join(result_folder, 'result_test_bad_gray.csv')
result_test_gray = os.path.join(result_folder, 'result_test_gray.csv')

##rgb files
history_rgb = os.path.join(os.getcwd(), 'models', 'rgb', 'history.csv')
result_test_good_rgb = os.path.join(result_folder, 'result_test_good_rgb.csv')
result_test_bad_rgb = os.path.join(result_folder, 'result_test_bad_rgb.csv')
result_test_rgb = os.path.join(result_folder, 'result_test_rgb.csv')

##Make Loss Curve Plots
loss_curve(history_rgb,
           history_gray)

##Make ROC Curve Plots
roc_curve(result_test_good_rgb,
          result_test_bad_rgb,
          result_test_good_gray,
          result_test_bad_gray
)

##Optimize the thesholds
optimize_scores(result_test_good_rgb,
                result_test_bad_rgb,
                result_test_good_gray,
                result_test_bad_gray
)

##Plot histograms of the test dataset scores
test_scores_plot(result_test_good_rgb, 
                 result_test_bad_rgb,
                 result_test_good_gray,
                 result_test_bad_gray)
