# -*- coding: utf-8 -*-
"""
@Time    : 2019/7/10 0:01
@Author  : shengty
"""
import numpy as np
import scipy.io as sio

import mne


def inputmat(fp):
    """load .mat file and return m as a dict"""
    mat = sio.loadmat(fp, squeeze_me=True)
    m = {}  # create a dict

    # Numpy array of size channel_num * points.
    m['data'] = mat['cnt'].T  # 数据
    m['freq'] = mat['nfo']['fs'][True][0]  # Sampling frequency

    # channel names are necessary information for creating a rawArray.
    m['ch_names'] = mat['nfo']['clab'][True][0]

    # Position of channels
    m['electrode_x'] = mat['nfo']['xpos'][True][0]
    m['electrode_y'] = mat['nfo']['ypos'][True][0]

    # find trials and its data
    m['cue'] = mat['mrk']['pos'][True][0]  # time of cue
    m['labels'] = np.nan_to_num(mat['mrk']['y'][True][0]).astype(int)  # convert NaN to 0
    m['n_trials'] = np.where(m['labels'] == 0)[0][0]  # Number of the total useful trials
    return m


def creatEventsArray(fp):
    """Create events array. The second column default to zero."""
    m = inputmat(fp)
    events = np.zeros((m['n_trials'], 3), int)
    events[:, 0] = m['cue'][:m['n_trials']]  # The first column is the sample number of the event.
    events[:, 2] = m['labels'][:m['n_trials']]  # The third column is the new event value.
    return events, m['labels']


def creatRawArray(fp):
    """Create a mne.io.RawArray object, data: array, shape (n_channels, n_times)"""
    m = inputmat(fp)
    ch_names = m['ch_names'].tolist()
    info = mne.create_info(ch_names, m['freq'], 'eeg')  # Create info for raw
    raw = mne.io.RawArray(m['data'], info, first_samp=0, copy='auto', verbose=None)
    return raw
