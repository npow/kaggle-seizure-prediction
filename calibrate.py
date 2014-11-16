import numpy as np
from common.data import CachedDataLoader
from sklearn.cross_validation import KFold
from sklearn.isotonic import *
from sklearn.linear_model import *
from sklearn.svm import *

cache_dir = 'data-cache'
cached_data_loader = CachedDataLoader(cache_dir)

def train_test_split(X, Y, cv_ratio, shuffle):
  indices = [(i,j) for (i,j) in KFold(X.shape[0], 2, shuffle=shuffle)]
  train_indices = indices[0][0]
  test_indices = indices[0][1]
  return X[train_indices], X[test_indices], Y[train_indices], Y[test_indices]

def flatten(data):
    if data.ndim > 2:
        return data.reshape((data.shape[0], np.product(data.shape[1:])))
    else:
        return data

TRAIN_FILES = [
    ['data_preictal_Dog_1_max-cross-corr', 'data_interictal_Dog_1_max-cross-corr'],
    ['data_preictal_Dog_2_max-cross-corr', 'data_interictal_Dog_2_max-cross-corr'],
    ['data_preictal_Dog_3_max-cross-corr', 'data_interictal_Dog_3_max-cross-corr'],
    ['data_preictal_Dog_4_max-cross-corr', 'data_interictal_Dog_4_max-cross-corr'],
    ['data_preictal_Dog_5_max-cross-corr', 'data_interictal_Dog_5_max-cross-corr'],
    ['data_preictal_Patient_1_max-cross-corr', 'data_interictal_Patient_1_max-cross-corr'],
    ['data_preictal_Patient_2_max-cross-corr', 'data_interictal_Patient_2_max-cross-corr']
]

PREICTAL_FILES = [x[0] for x in TRAIN_FILES]
INTERICTAL_FILES = [x[1] for x in TRAIN_FILES]

TEST_FILES = [
    'data_test_Dog_1_max-cross-corr',
    'data_test_Dog_2_max-cross-corr',
    'data_test_Dog_3_max-cross-corr',
    'data_test_Dog_4_max-cross-corr',
    'data_test_Dog_5_max-cross-corr',
    'data_test_Patient_1_max-cross-corr',
    'data_test_Patient_2_max-cross-corr'
]

PREFIXES = ['Dog_1', 'Dog_2', 'Dog_3', 'Dog_4', 'Dog_5', 'Patient_1', 'Patient_2']
FILENAMES = []

TRAIN_PROBAS = []
TRAIN_LABELS = []
TEST_PROBAS = []
for i in range(len(TRAIN_FILES)):
    preictal = cached_data_loader.load(PREICTAL_FILES[i], None)
    preictal_X = flatten(preictal.__dict__['X'])
    preictal_Y = flatten(preictal.__dict__['y'])
    interictal = cached_data_loader.load(INTERICTAL_FILES[i], None)
    interictal_X = flatten(interictal.__dict__['X'])
    interictal_Y = flatten(interictal.__dict__['y'])

    preictal_x_train, preictal_x_test, preictal_y_train, preictal_y_test = train_test_split(preictal_X, preictal_Y, cv_ratio=0.5, shuffle=False)
    interictal_x_train, interictal_x_test, interictal_y_train, interictal_y_test = train_test_split(interictal_X, interictal_Y, cv_ratio=0.5, shuffle=False)
    x_train = np.concatenate([preictal_x_train, interictal_x_train])
    y_train = np.concatenate([preictal_y_train, interictal_y_train])
    print "x_train: ", x_train.shape
    print "y_train: ", y_train.shape

    x_test = np.concatenate([preictal_x_test, interictal_x_test])
    y_test = np.concatenate([preictal_y_test, interictal_y_test])
    print "x_test: ", x_test.shape
    print "y_test: ", y_test.shape

    clf = SVC(C=1000, probability=True)
    clf.fit(x_train, y_train)
    train_probas = clf.predict_proba(x_test)[:,1]
    TRAIN_PROBAS.append(train_probas)
    TRAIN_LABELS.append(y_test)

    X_train = np.concatenate([preictal_X, interictal_X])
    Y_train = np.concatenate([preictal_Y, interictal_Y])
    test_data = cached_data_loader.load(TEST_FILES[i], None)
    X_test = flatten(test_data.__dict__['X'])
    print "X_test: ", X_test.shape

    svc = SVC(C=1000, probability=True)
    svc.fit(X_train, Y_train)
    test_probas = svc.predict_proba(X_test)[:,1]
    print "test_probas: ", test_probas.shape
    TEST_PROBAS.append(test_probas)
    for j in xrange(X_test.shape[0]):
        FILENAMES.append("%s_test_segment_%04d.mat" % (PREFIXES[i], j+1))

TRAIN_PROBAS = np.concatenate(TRAIN_PROBAS)
TRAIN_LABELS = np.concatenate(TRAIN_LABELS)
TEST_PROBAS = np.concatenate(TEST_PROBAS)
print "TRAIN_PROBAS: ", TRAIN_PROBAS.reshape(-1, 1).shape
print "TRAIN_LABELS: ", TRAIN_LABELS.shape

"""
lr = LogisticRegression()
lr.fit(TRAIN_PROBAS.reshape(-1, 1), TRAIN_LABELS)
calibrated_probas = lr.predict_proba(TEST_PROBAS.reshape(-1, 1))[:,1]
"""
ir = IsotonicRegression(out_of_bounds='clip')
ir.fit(TRAIN_PROBAS, TRAIN_LABELS)
calibrated_probas = ir.transform(TEST_PROBAS)
print "calibrated_probas: ", calibrated_probas.shape

f = open('combined_ir_c1000.csv', 'wb')
f.write('clip,preictal\n')
for idx, p in enumerate(calibrated_probas):
    f.write('%s,%f\n' % (FILENAMES[idx], p))
f.close()
