#!/usr/bin/env python

import os, sys, pickle, itertools
from math import *
from numpy import *
from scipy.io import loadmat
from scipy.signal import resample, butter, lfilter

from sklearn.externals import joblib
import sklearn.linear_model
import sklearn.cross_validation
import sklearn.metrics

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Prepare 8 overlapping log-spaced bandpass filters between 5 and 200Hz
filters = []
for i in range(8):
    x = logspace(log10(5),log10(200),10)
    filters.append(butter_bandpass(x[i], floor(x[i+2]),400,3))

# List of all possible combinations of 4 or fewer filters
wis = list(itertools.chain(itertools.combinations(range(8),1),itertools.combinations(range(8),2),itertools.combinations(range(8),3),itertools.combinations(range(8),4)))

# Labels to read and write data for each subject
inlabels = ['Dog_1','Dog_2','Dog_3','Dog_4', 'Dog_5', 'Patient_1','Patient_2']
outlabels = ['dog_0', 'dog_1', 'dog_2', 'dog_3', 'dog_4', 'patient_0', 'patient_1']

def process(subject):
    """
        Given a subject index, filter each clip, calculate covariance matrices, 
        calculate top-3 filter sets, and save processed data to pickle.
    """
    dn = './data/clips/%s/'%inlabels[subject]
    fns = [fn for fn in os.listdir(dn) if '.mat' in fn]

    allcovs = []
    labels = []

    print dn
    # For each clip, resample to 400Hz, apply each filter, calculate and normalize covariance.
    for fn in fns:
        covs = []   
        m = loadmat(dn+fn)
        d = m['data']
        d = resample(d, 400, axis=1)
        if 'inter' in fn:
            l = 0
        elif '_ictal' in fn:
            l = 1
        else:
            l = -1

        labels.append(l)

        for b, a in filters:
            f = lfilter(b,a,d)
            c = cov(f)
            c = (c-c.mean())/c.std()
            covs.append(c)
        allcovs.append(covs)
    allcovs = array(allcovs)
    labels = array(labels)
    
    # For each filter combination, test prediction quality by CV of logistic regression.
    scores = []
    for w in wis:
        y = labels[labels != -1]
        X = allcovs[labels != -1]
        X = X[:,w,::2,::2]
        X = X.reshape((X.shape[0],-1))

        ps = []
        test_size = 0.25
        for tri, tei in sklearn.cross_validation.ShuffleSplit(X.shape[0], n_iter=15, test_size=test_size, random_state=42):
            X_train = X[tri]
            X_test = X[tei]
            y_train = y[tri]
            y_test = y[tei]

            clf = sklearn.linear_model.SGDClassifier(loss='log', penalty='l1', alpha=0.0001)
            clf.fit(X_train, y_train)
            p = clf.predict_proba(X_test)[:,1]
            cv = sklearn.metrics.roc_auc_score(y_test, p)
            ps.append(cv)
        ps = array(ps)
        scores.append(ps.mean())    

    # Select 3 best filter sets and save processed features and labels to pickle.
    best = sorted(zip(scores,wis))[-3:]
    sets = 'ABC'
    i = 0
    for cv, w in best:
        print outlabels[subject], cv, w
        y = labels
        X = allcovs
        X = X[:,w,:,:]
        d = {'y':y, 'covs':X, 'w':w, 'cv':cv, 'fns':fns}
        pickle.dump(d, open('./data/cov_opt_%s_%s.pickle'%(outlabels[subject],sets[i]), 'w'))
        i += 1

# Process all subjects in parallel
# reduce n_jobs if out of memory
r = joblib.Parallel(n_jobs=-1)(joblib.delayed(process)(n) for n in range(12))
