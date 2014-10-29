import json
import os.path
import numpy as np
from common import time
from common.data import CachedDataLoader, makedirs
from common.pipeline import Pipeline
from seizure.transforms import *
from seizure.tasks import TaskCore, CrossValidationScoreTask, MakePredictionsTask, TrainClassifierTask
from seizure.scores import get_score_summary, print_results

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

def run_seizure_detection(build_target):
    """
    The main entry point for running seizure-detection cross-validation and predictions.
    Directories from settings file are configured, classifiers are chosen, pipelines are
    chosen, and the chosen build_target ('cv', 'predict', 'train_model') is run across
    all combinations of (targets, pipelines, classifiers)
    """

    with open('SETTINGS.json') as f:
        settings = json.load(f)

    data_dir = str(settings['competition-data-dir'])
    cache_dir = str(settings['data-cache-dir'])
    submission_dir = str(settings['submission-dir'])

    makedirs(submission_dir)

    cached_data_loader = CachedDataLoader(cache_dir)

    ts = time.get_millis()

    targets = [
        #'Dog_1',
        #'Dog_2',
        #'Dog_3',
        #'Dog_4',
        'Dog_5',
        #'Patient_1',
        #'Patient_2'
    ]
    pipelines = [
        # NOTE(mike): you can enable multiple pipelines to run them all and compare results
        Pipeline(gen_preictal=False, pipeline=[Resample(400), MaximalCrossCorrelation()]),
        #Pipeline(gen_preictal=False, pipeline=[CorrelationWithVariance(with_eigen=False)]),
        #Pipeline(gen_preictal=True, pipeline=[CorrelationWithVariance(with_eigen=True)]),
        #Pipeline(gen_preictal=True, pipeline=[CorrelationWithVariance(with_eigen=False)]),
        #Pipeline(gen_preictal=True,  pipeline=[FFT(), Slice(1, 48), Magnitude(), Log10()]),
        #Pipeline(gen_preictal=False, pipeline=[MFCC()]),
        #Pipeline(gen_preictal=False, pipeline=[CorrelationWithVariance()]),
        #Pipeline(gen_preictal=False, pipeline=[FFT(), Slice(1, 64), Magnitude(), Log10()]),
        #Pipeline(gen_preictal=False, pipeline=[FFT(), Slice(1, 96), Magnitude(), Log10()]),
        #Pipeline(gen_preictal=False, pipeline=[FFT(), Slice(1, 128), Magnitude(), Log10()]),
        #Pipeline(gen_preictal=False, pipeline=[FFT(), Slice(1, 160), Magnitude(), Log10()]),
        #Pipeline(gen_preictal=False, pipeline=[FFT()]),
        #Pipeline(gen_preictal=False, pipeline=[FFT(), Magnitude(), Log10()]),
        #Pipeline(gen_preictal=False, pipeline=[Stats()]),
        #Pipeline(gen_preictal=False, pipeline=[DaubWaveletStats(4)]),
        #Pipeline(gen_preictal=False, pipeline=[Resample(400), DaubWaveletStats(4)]),
        #Pipeline(gen_preictal=False, pipeline=[Resample(400), MFCC()]),
        #Pipeline(gen_preictal=False, pipeline=[FFTWithTimeFreqCorrelation(65, 100, 400, 'us')]),
        #Pipeline(gen_preictal=False, pipeline=[FFTWithTimeFreqCorrelation(30, 45, 400, 'us')]),
        #Pipeline(gen_preictal=False, pipeline=[FFTWithTimeFreqCorrelation(1, 48, 400, 'us')]),
        #Pipeline(gen_preictal=True,  pipeline=[FFTWithTimeFreqCorrelation(1, 48, 400, 'us')]),
        #Pipeline(gen_preictal=True,  pipeline=[FFTWithTimeFreqCorrelation(1, 48, 400, 'usf')]), # winning submission
        #Pipeline(gen_preictal=True,  pipeline=[FFTWithTimeFreqCorrelation(1, 48, 400, 'usf')]), # higher score than winning submission
        #Pipeline(gen_preictal=False, pipeline=[FFTWithTimeFreqCorrelation(1, 48, 400, 'none')]),
        #Pipeline(gen_preictal=True,  pipeline=[FFTWithTimeFreqCorrelation(1, 48, 400, 'none')]),
        #Pipeline(gen_preictal=False, pipeline=[TimeCorrelation(400, 'usf', with_corr=True, with_eigen=True)]),
        #Pipeline(gen_preictal=False, pipeline=[TimeCorrelation(400, 'us', with_corr=True, with_eigen=True)]),
        #Pipeline(gen_preictal=False, pipeline=[TimeCorrelation(400, 'us', with_corr=True, with_eigen=False)]),
        #Pipeline(gen_preictal=False, pipeline=[TimeCorrelation(400, 'us', with_corr=False, with_eigen=True)]),
        #Pipeline(gen_preictal=False, pipeline=[TimeCorrelation(400, 'none', with_corr=True, with_eigen=True)]),
        #Pipeline(gen_preictal=False, pipeline=[FreqCorrelation(1, 48, 'usf', with_corr=True, with_eigen=True)]),
        #Pipeline(gen_preictal=False, pipeline=[FreqCorrelation(1, 48, 'us', with_corr=True, with_eigen=True)]),
        #Pipeline(gen_preictal=False, pipeline=[FreqCorrelation(1, 48, 'us', with_corr=True, with_eigen=False)]),
        #Pipeline(gen_preictal=False, pipeline=[FreqCorrelation(1, 48, 'us', with_corr=False, with_eigen=True)]),
        #Pipeline(gen_preictal=False, pipeline=[FreqCorrelation(1, 48, 'none', with_corr=True, with_eigen=True)]),
        #Pipeline(gen_preictal=False, pipeline=[TimeFreqCorrelation(1, 48, 400, 'us')]),
        #Pipeline(gen_preictal=False, pipeline=[TimeFreqCorrelation(1, 48, 400, 'usf')]),
        #Pipeline(gen_preictal=False, pipeline=[TimeFreqCorrelation(1, 48, 400, 'none')]),
    ]
    classifiers = [
        # NOTE(mike): you can enable multiple classifiers to run them all and compare results
        (RandomForestClassifier(n_estimators=50, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf50mss1Bfrs0'),
        #(RandomForestClassifier(n_estimators=150, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf150mss1Bfrs0'),
        #(RandomForestClassifier(n_estimators=300, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf300mss1Bfrs0'),
        #(RandomForestClassifier(n_estimators=1000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf1000mss1Bfrs0'),
        #(RandomForestClassifier(n_estimators=2000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf2000mss1Bfrs0'),
        #(RandomForestClassifier(n_estimators=3000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf3000mss1Bfrs0'),
        #(RandomForestClassifier(n_estimators=4000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf4000mss1Bfrs0'),
        #(RandomForestClassifier(n_estimators=5000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf5000mss1Bfrs0'),
        #(RandomForestClassifier(n_estimators=6000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf6000mss1Bfrs0'),
        #(RandomForestClassifier(n_estimators=7000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf7000mss1Bfrs0'),
        #(RandomForestClassifier(n_estimators=8000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf8000mss1Bfrs0'),
        #(RandomForestClassifier(n_estimators=10000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf10000mss1Bfrs0'),
        #(RandomForestClassifier(n_estimators=9000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf9000mss1Bfrs0'),
        #(RandomForestClassifier(n_estimators=11000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf11000mss1Bfrs0'),
        #(RandomForestClassifier(n_estimators=12000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf12000mss1Bfrs0'),
        #(RandomForestClassifier(n_estimators=13000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf13000mss1Bfrs0'),
        #(RandomForestClassifier(n_estimators=14000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf14000mss1Bfrs0'),
        #(RandomForestClassifier(n_estimators=15000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf15000mss1Bfrs0'),
        #(RandomForestClassifier(n_estimators=16000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf16000mss1Bfrs0'),
        #(RandomForestClassifier(n_estimators=17000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf17000mss1Bfrs0'),
        #(RandomForestClassifier(n_estimators=18000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf18000mss1Bfrs0'),
        #(RandomForestClassifier(n_estimators=19000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf19000mss1Bfrs0'),
        #(RandomForestClassifier(n_estimators=20000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0), 'rf20000mss1Bfrs0'),
        #(LogisticRegression(), 'logistic_regression'),
        #(LinearSVC(C=0.1), 'linearsvc_c0.1'),
        #(LinearSVC(C=1), 'linearsvc_c1'),
    ]
    cv_ratio = 0.5

    def should_normalize(classifier):
        clazzes = [LogisticRegression]
        return np.any(np.array([isinstance(classifier, clazz) for clazz in clazzes]) == True)

    def train_full_model(make_predictions):
        for pipeline in pipelines:
            for (classifier, classifier_name) in classifiers:
                print 'Using pipeline %s with classifier %s' % (pipeline.get_name(), classifier_name)
                guesses = ['clip,preictal']
                classifier_filenames = []
                for target in targets:
                    task_core = TaskCore(cached_data_loader=cached_data_loader, data_dir=data_dir,
                                         target=target, pipeline=pipeline,
                                         classifier_name=classifier_name, classifier=classifier,
                                         normalize=should_normalize(classifier), gen_preictal=pipeline.gen_preictal,
                                         cv_ratio=cv_ratio)

                    if make_predictions:
                        predictions = MakePredictionsTask(task_core).run()
                        guesses.append(predictions.data)
                    else:
                        task = TrainClassifierTask(task_core)
                        task.run()
                        classifier_filenames.append(task.filename())

                if make_predictions:
                    filename = 'submission%d-%s_%s.csv' % (ts, classifier_name, pipeline.get_name())
                    filename = os.path.join(submission_dir, filename)
                    with open(filename, 'w') as f:
                        print >> f, '\n'.join(guesses)
                    print 'wrote', filename
                else:
                    print 'Trained classifiers ready in %s' % cache_dir
                    for filename in classifier_filenames:
                        print os.path.join(cache_dir, filename + '.pickle')

    def do_cross_validation():
        summaries = []
        for pipeline in pipelines:
            for (classifier, classifier_name) in classifiers:
                print 'Using pipeline %s with classifier %s' % (pipeline.get_name(), classifier_name)
                scores = []
                for target in targets:
                    print 'Processing %s (classifier %s)' % (target, classifier_name)

                    task_core = TaskCore(cached_data_loader=cached_data_loader, data_dir=data_dir,
                                         target=target, pipeline=pipeline,
                                         classifier_name=classifier_name, classifier=classifier,
                                         normalize=should_normalize(classifier), gen_preictal=pipeline.gen_preictal,
                                         cv_ratio=cv_ratio)

                    data = CrossValidationScoreTask(task_core).run()
                    score = data.score

                    scores.append(score)

                    print '%.3f' % score

                if len(scores) > 0:
                    name = pipeline.get_name() + '_' + classifier_name
                    summary = get_score_summary(name, scores)
                    summaries.append((summary, np.mean(scores)))
                    print summary

            print_results(summaries)

    if build_target == 'cv':
        do_cross_validation()
    elif build_target == 'train_model':
        train_full_model(make_predictions=False)
    elif build_target == 'make_predictions':
        train_full_model(make_predictions=True)
    else:
        raise Exception("unknown build target %s" % build_target)
