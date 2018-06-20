'''Brain decoding dataset.'''


from abc import ABCMeta, abstractmethod
import os


class DatasetBase(object):
    '''Dataset class.

    Attributes
    ----------
    datastore : str, optional
        Path to data store directory
    verbose : bool, optional
        Output verbose messages or not
    '''

    __metaclass__ = ABCMeta

    def __init__(self, datastore=None, verbose=False):
        self._verbose = verbose

        if datastore is None:
            self._datastore = os.path.join(os.environ['HOME'], '.bdds')
        else:
            self._datastore = datastore
        if self._verbose:
            print('Initialize dataset.')
            print('Data store: %s' % self._datastore)

    def get(self, **kargs):
        '''Returns specified data.'''

        # Return value settings
        return_dict = kargs['return_dict'] if 'return_dict' in kargs else False
        force_list = kargs['force_list'] if 'force_list' in kargs else False
        if 'return_dict' in kargs: del(kargs['return_dict'])
        if 'force_list' in kargs: del(kargs['force_list'])

        if self._verbose:
            if return_dict: print('return_dict enabled.')
            if force_list:  print('force_list enabled.')
        
        # Get data
        collection = self._get_files(**kargs)
        collection = self.__load_data(collection)

        # Postprocessing
        if return_dict:
            output = collection
        else:
            output = [item['data'] for item in collection]

        if len(output) == 1 and not force_list:
            output = output[0]

        return output

    def __load_data(self, collection):
        '''Load data.

        `__load_data` tries to load data from local storage. If the file is
        missing, it downloads the file from remote repository.
        '''

        for item in collection:
            fpath = os.path.join(self._datastore, item['file'])

            if not os.path.exists(fpath):
                if self._verbose: print('Downloading file %s' % item['file'])
                self._download_file(item['file'])

            if self._verbose: print('Loading %s' % fpath)
            dat = self._load_file(fpath)
            item['data'] = dat

        return collection

    @abstractmethod
    def _get_files(self, **kargs):
        pass

    @abstractmethod
    def _load_file(self, fpath):
        pass

    @abstractmethod
    def _download_file(self, fname):
        pass
