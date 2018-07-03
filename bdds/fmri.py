'''fMRI datasets.'''


__all__ = ['HandShapeDecoding']


import os
import urllib

import bdpy

from .bdds import DatasetBase


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

    def __init__(self, datastore=None, verbose=False):
        super(HandShapeDecoding, self).__init__(datastore=datastore, verbose=verbose)
        # Default data store path
        if datastore is None:
            self._datastore = os.path.join(self._datastore, 'handshape')

        if not os.path.exists(self._datastore):
            os.makedirs(self._datastore)

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

        # Download file
        print('Downloading from %s' % url)
        urllib.urlretrieve(url,
                           os.path.join(self._datastore, fname))
        print('Saved %s' % os.path.join(self._datastore, fname))
        return None

    def __listize(self, x):
        return x if isinstance(x, list) else [x]
