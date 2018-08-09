'''fMRI datasets.'''


__all__ = ['HandShapeDecoding']


import os

import bdpy

from .bdds import DatasetBase
from .download import download_file


class HandShapeDecoding(DatasetBase):
    '''Hand shape decoding dataset class.

    Attributes
    ----------
    datastore : str, optional
        Path to data store directory
    verbose : bool, optional
        Output verbose messages or not
    '''

    __modes = ['fmri']
    __subjects = ['S1']

    __remote_files = {'S1.h5': 'https://ndownloader.figshare.com/files/12227786'}

    @property
    def _remote_files(self):
        return self.__remote_files

    def __init__(self, datastore=None, verbose=False, auto_download=False):
        super(HandShapeDecoding, self).__init__(datastore=datastore, verbose=verbose, auto_download=auto_download, default_dir='handshape')

    def _get_files(self, mode='fmri', subject='S1'):

        if subject is None: subject = HandShapeDecoding.__subjects
        subject = self.__listize(subject)

        # Disp info
        if self._verbose:
            print('Mode :        %s' % mode)
            print('Subject :     %s' % subject)

        # TODO: add args value check

        # Get data files
        collection = []

        for sbj in subject:
            fpath = sbj + '.h5'
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
        url = HandShapeDecoding.__remote_files[fname]

        download_file(url, os.path.join(self._datastore, fname))

        return None

    def __listize(self, x):
        return x if isinstance(x, list) else [x]
