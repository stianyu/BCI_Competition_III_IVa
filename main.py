# -*- coding: utf-8 -*-
"""
@Time    : 2019/7/10 10:14
@Author  : shengty
"""
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from mne.decoding import CSP
from mne.channels import read_layout
from loaddata import *

print(__doc__)

fp = {
    'aa': './data_set_raw/data_set_IVa_aa.mat',
    'al': './data_set_raw/data_set_IVa_al.mat',
    'av': './data_set_raw/data_set_IVa_av.mat',
    'aw': './data_set_raw/data_set_IVa_aw.mat',
    'ay': './data_set_raw/data_set_IVa_ay.mat',
}
pick_chan = {
    'aa': ['C3', 'Cz', 'C5'],
    'al': ['C3', 'Cz', 'C5'],
    'av': ['C3', 'Cz', 'C5'],
    'aw': ['C3', 'Cz', 'C5'],
    'ay': ['C3', 'Cz', 'C5'],
}

low_freq, high_freq = 7., 30.
tmin, tmax = 0., 3.5

# event_id
event_id = {'right': 1, 'foot': 2}

for f in fp:
    raw = creatRawArray(fp[f])
    events, labels = creatEventsArray(fp[f])

    # Apply band-pass filter
    raw.filter(low_freq, high_freq, fir_design='firwin', skip_by_annotation='edge')

    # event_train = eventsTrain(fp[f])
    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True,
                        verbose=False)

    epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    labels = epochs.events[:, -1] - 2

    # Define a monte-carlo cross-validation generator (reduce variance):
    scores = []
    epochs_data = epochs.get_data()
    epochs_data_train = epochs_train.get_data()
    cv = ShuffleSplit(5, test_size=0.2, random_state=42)
    cv_split = cv.split(epochs_data_train)

    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=len(epochs.ch_names), reg=None, log=True, norm_trace=False)

    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)

    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores), class_balance))

    # plot CSP patterns estimated on full data for visualization
    csp.fit_transform(epochs_data, labels)

    # layout = read_layout('EEG1005')
    # csp.plot_patterns(epochs.info, layout=layout, ch_type='eeg', units='Patterns (AU)', size=1.5)
