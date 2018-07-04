'''Generic object decoding dataset.'''


__all__ = ['GenericObjectDecoding']


import os
import warnings
from itertools import product
from collections import OrderedDict
import urllib
import zipfile
import shutil
import datetime

import numpy as np
import scipy.io as sio
import h5py

import bdpy

from .bdds import DatasetBase


class GenericObjectDecoding(DatasetBase):
    '''Generic object decoding dataset class.

    Attributes
    ----------
    datastore : str, optional
        Path to data store directory
    verbose : bool, optional
        Output verbose messages or not
    '''

    __modes = ['fmri', 'image_features']
    __subjects = ['Subject1',
                  'Subject2',
                  'Subject3',
                  'Subject4',
                  'Subject5']

    __remote_files = {'Subject1.mat': 'http://brainliner.jp/download/32/downloadSupplementaryFile',
                      'Subject2.mat': 'http://brainliner.jp/download/36/downloadSupplementaryFile',
                      'Subject3.mat': 'http://brainliner.jp/download/34/downloadSupplementaryFile',
                      'Subject4.mat': 'http://brainliner.jp/download/35/downloadSupplementaryFile',
                      'Subject5.mat': 'http://brainliner.jp/download/33/downloadSupplementaryFile',
                      'ImageFeatures.h5': 'http://brainliner.jp/download/1332/downloadDataFile'}

    def __init__(self, datastore=None, verbose=False):
        super(GenericObjectDecoding, self).__init__(datastore=datastore, verbose=verbose, default_dir='god')

    def _get_files(self, mode=None, subject=None):

        # Input processing
        if mode is None: raise RuntimeError('`mode` is required.')

        if subject is None: subject = GenericObjectDecoding.__subjects
        subject = self.__listize(subject)

        # Disp info
        if self._verbose:
            print('Mode :        %s' % mode)
            print('Subject :     %s' % subject)

        # TODO: add args value check

        # Get data files
        collection = []

        if mode == 'image_features':
            fpath = 'ImageFeatures.h5'
            collection.append({'identifier': {'mode': mode,
                                              'subject': None},
                                   'file': fpath,
                                   'data': None})
        else:
            for sbj in subject:
                fpath = sbj + '.mat'
                collection.append({'identifier': {'mode': mode,
                                                  'subject': sbj},
                                   'file': fpath,
                                   'data': None})

        return collection

    def _load_file(self, fpath):
        try:
            return bdpy.BData(fpath)
        except:
            raise RuntimeError('Invalid data: %s' % fpath)

    def _download_file(self, fname):
        url = GenericObjectDecoding.__remote_files[fname]

        # Download file
        print('Downloading from %s' % url)
        urllib.urlretrieve(url,
                           os.path.join(self._datastore, fname))
        return None

    def __listize(self, x):
        return x if isinstance(x, list) else [x]
