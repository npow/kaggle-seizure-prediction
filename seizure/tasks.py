from collections import namedtuple
import os.path
import numpy as np
import scipy.io
import common.time as time
from sklearn import cross_validation, preprocessing
from sklearn.metrics import roc_curve, auc

TaskCore = namedtuple('TaskCore', ['cached_data_loader', 'data_dir', 'target', 'pipeline', 'classifier_name',
                                   'classifier', 'normalize', 'gen_preictal', 'cv_ratio'])

class Task(object):
    """
    A Task computes some work and outputs a dictionary which will be cached on disk.
    If the work has been computed before and is present in the cache, the data will
    simply be loaded from disk and will not be pre-computed.
    """
    def __init__(self, task_core):
        self.task_core = task_core

    def filename(self):
        raise NotImplementedError("Implement this")

    def run(self):
        return self.task_core.cached_data_loader.load(self.filename(), self.load_data)


class LoadPreictalDataTask(Task):
    """
    Load the preictal mat files 1 by 1, transform each 1-second segment through the pipeline
    and return data in the format {'X': X, 'Y': y}
    """
    def filename(self):
        return 'data_preictal_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name())

    def load_data(self):
        return parse_input_data(self.task_core.data_dir, self.task_core.target, 'preictal', self.task_core.pipeline,
                           self.task_core.gen_preictal)


class LoadInterictalDataTask(Task):
    """
    Load the interictal mat files 1 by 1, transform each 1-second segment through the pipeline
    and return data in the format {'X': X, 'Y': y}
    """
    def filename(self):
        return 'data_interictal_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name())

    def load_data(self):
        return parse_input_data(self.task_core.data_dir, self.task_core.target, 'interictal', self.task_core.pipeline)


class LoadTestDataTask(Task):
    """
    Load the test mat files 1 by 1, transform each 1-second segment through the pipeline
    and return data in the format {'X': X}
    """
    def filename(self):
        return 'data_test_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name())

    def load_data(self):
        return parse_input_data(self.task_core.data_dir, self.task_core.target, 'test', self.task_core.pipeline)


class TrainingDataTask(Task):
    """
    Creating a training set and cross-validation set from the transformed preictal and interictal data.
    """
    def filename(self):
        return None  # not cached, should be fast enough to not need caching

    def load_data(self):
        preictal_data = LoadPreictalDataTask(self.task_core).run()
        interictal_data = LoadInterictalDataTask(self.task_core).run()
        return prepare_training_data(preictal_data, interictal_data, self.task_core.cv_ratio)


class CrossValidationScoreTask(Task):
    """
    Run a classifier over a training set, and give a cross-validation score.
    """
    def filename(self):
        return 'score_%s_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name(), self.task_core.classifier_name)

    def load_data(self):
        data = TrainingDataTask(self.task_core).run()
        classifier_data = train_classifier(self.task_core.classifier, data, normalize=self.task_core.normalize)
        del classifier_data['classifier'] # save disk space
        return classifier_data


class TrainClassifierTask(Task):
    """
    Run a classifier over the complete data set (training data + cross-validation data combined)
    and save the trained models.
    """
    def filename(self):
        return 'classifier_%s_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name(), self.task_core.classifier_name)

    def load_data(self):
        data = TrainingDataTask(self.task_core).run()
        return train_classifier(self.task_core.classifier, data, use_all_data=True, normalize=self.task_core.normalize)


class MakePredictionsTask(Task):
    """
    Make predictions on the test data.
    """
    def filename(self):
        return 'predictions_%s_%s_%s' % (self.task_core.target, self.task_core.pipeline.get_name(), self.task_core.classifier_name)

    def load_data(self):
        data = TrainingDataTask(self.task_core).run()
        y_classes = data.y_classes
        del data

        classifier_data = TrainClassifierTask(self.task_core).run()
        test_data = LoadTestDataTask(self.task_core).run()
        X_test = flatten(test_data.X)

        return make_predictions(self.task_core.target, X_test, y_classes, classifier_data)

#generator to iterate over competition mat data
def load_mat_data(data_dir, target, component):
    dir = os.path.join(data_dir, target)
    done = False
    i = 0
    while not done:
        i += 1
        filename = '%s/%s_%s_segment_%d.mat' % (dir, target, component, i)
        if os.path.exists(filename):
            data = scipy.io.loadmat(filename)
            yield(data)
        else:
            if i == 1:
                raise Exception("file %s not found" % filename)
            done = True


# process all of one type of the competition mat data
# data_type is one of ('preictal', 'interictal', 'test')
def parse_input_data(data_dir, target, data_type, pipeline, gen_preictal=False):
    preictal = data_type == 'preictal'
    interictal = data_type == 'interictal'

    mat_data = load_mat_data(data_dir, target, data_type)

    # for each data point in preictal, interictal and test,
    # generate (X, <y>) per channel
    def process_raw_data(mat_data):
        start = time.get_seconds()
        print 'Loading data',
        X = []
        y = []

        prev_data = None
        for segment in mat_data:
            data = segment['data']
            transformed_data = pipeline.apply(data)

            ### WTF IS THIS?
            y.append(2)

            X.append(transformed_data)
            prev_data = data

        print '(%ds)' % (time.get_seconds() - start)

        X = np.array(X)
        y = np.array(y)

        if preictal:
            print 'X', X.shape, 'y', y.shape
            return X, y
        elif interictal:
            print 'X', X.shape, 'y', y.shape
            return X, y
        else:
            print 'X', X.shape
            return X

    data = process_raw_data(mat_data)

    if len(data) == 2:
        X, y = data
        return {
            'X': X,
            'y': y
        }
    else:
        X = data
        return {
            'X': X
        }


# flatten data down to 2 dimensions for putting through a classifier
def flatten(data):
    if data.ndim > 2:
        return data.reshape((data.shape[0], np.product(data.shape[1:])))
    else:
        return data


# split up preictal and interictal data into training set and cross-validation set
def prepare_training_data(ictal_data, interictal_data, cv_ratio):
    print 'Preparing training data ...',
    preictal_X, preictal_y = flatten(preictal_data.X), preictal_data.y
    interictal_X, interictal_y = flatten(interictal_data.X), interictal_data.y

    # split up data into training set and cross-validation set for both seizure and early sets
    preictal_X_train, preictal_y_train, preictal_X_cv, preictal_y_cv = train_test_split(ictal_X, ictal_y, cv_ratio)
    interictal_X_train, interictal_y_train, interictal_X_cv, interictal_y_cv = split_train_random(interictal_X, interictal_y, cv_ratio)

    def concat(a, b):
        return np.concatenate((a, b), axis=0)

    X_train = concat(preictal_X_train, interictal_X_train)
    y_train = concat(preictal_y_train, interictal_y_train)
    X_cv = concat(preictal_X_cv, interictal_X_cv)
    y_cv = concat(preictal_y_cv, interictal_y_cv)

    y_classes = np.unique(concat(y_train, y_cv))

    start = time.get_seconds()
    elapsedSecs = time.get_seconds() - start
    print "%ds" % int(elapsedSecs)

    print 'X_train:', np.shape(X_train)
    print 'y_train:', np.shape(y_train)
    print 'X_cv:', np.shape(X_cv)
    print 'y_cv:', np.shape(y_cv)
    print 'y_classes:', y_classes

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_cv': X_cv,
        'y_cv': y_cv,
        'y_classes': y_classes
    }


# split interictal segments at random for training and cross-validation
def split_train_random(X, y, cv_ratio):
    X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X, y, test_size=cv_ratio, random_state=0)
    return X_train, y_train, X_cv, y_cv

# train classifier for cross-validation
def train(classifier, X_train, y_train, X_cv, y_cv, y_classes):
    print "Training ..."

    print 'Dim', 'X', np.shape(X_train), 'y', np.shape(y_train), 'X_cv', np.shape(X_cv), 'y_cv', np.shape(y_cv)
    start = time.get_seconds()
    classifier.fit(X_train, y_train)
    print "Scoring..."
    S, E = score_classifier_auc(classifier, X_cv, y_cv, y_classes)
    score = 0.5 * (S + E)

    elapsedSecs = time.get_seconds() - start
    print "t=%ds score=%f" % (int(elapsedSecs), score)
    return score, S, E


# train classifier for predictions
def train_all_data(classifier, X_train, y_train, X_cv, y_cv):
    print "Training ..."
    X = np.concatenate((X_train, X_cv), axis=0)
    y = np.concatenate((y_train, y_cv), axis=0)
    print 'Dim', np.shape(X), np.shape(y)
    start = time.get_seconds()
    classifier.fit(X, y)
    elapsedSecs = time.get_seconds() - start
    print "t=%ds" % int(elapsedSecs)


# sub mean divide by standard deviation
def normalize_data(X_train, X_cv):
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_cv = scaler.transform(X_cv)

    return X_train, X_cv

# depending on input train either for predictions or for cross-validation
def train_classifier(classifier, data, use_all_data=False, normalize=False):
    X_train = data.X_train
    y_train = data.y_train
    X_cv = data.X_cv
    y_cv = data.y_cv

    if normalize:
        X_train, X_cv = normalize_data(X_train, X_cv)

    if not use_all_data:
        score, S, E = train(classifier, X_train, y_train, X_cv, y_cv, data.y_classes)
        return {
            'classifier': classifier,
            'score': score,
            'S_auc': S,
            'E_auc': E
        }
    else:
        train_all_data(classifier, X_train, y_train, X_cv, y_cv)
        return {
            'classifier': classifier
        }


# convert the output of classifier predictions into (Seizure, Early) pair
def translate_prediction(prediction, y_classes):
  raise NotImplementedError()


# use the classifier and make predictions on the test data
def make_predictions(target, X_test, y_classes, classifier_data):
    classifier = classifier_data.classifier
    predictions_proba = classifier.predict_proba(X_test)

    lines = []
    for i in range(len(predictions_proba)):
        p = predictions_proba[i]
        S, E = translate_prediction(p, y_classes)
        lines.append('%s_test_segment_%d.mat,%.15f,%.15f' % (target, i+1, S, E))

    return {
        'data': '\n'.join(lines)
    }


# the scoring mechanism used by the competition leaderboard
def score_classifier_auc(classifier, X_cv, y_cv, y_classes):
    predictions = classifier.predict_proba(X_cv)
    S_predictions = []
    E_predictions = []
    S_y_cv = [1.0 if (x == 0.0 or x == 1.0) else 0.0 for x in y_cv]
    E_y_cv = [1.0 if x == 0.0 else 0.0 for x in y_cv]

    for i in range(len(predictions)):
        p = predictions[i]
        S, E = translate_prediction(p, y_classes)
        S_predictions.append(S)
        E_predictions.append(E)

    fpr, tpr, thresholds = roc_curve(S_y_cv, S_predictions)
    S_roc_auc = auc(fpr, tpr)
    fpr, tpr, thresholds = roc_curve(E_y_cv, E_predictions)
    E_roc_auc = auc(fpr, tpr)

    return S_roc_auc, E_roc_auc

