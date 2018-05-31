'''Brain decoding dataset.'''


from abc import ABCMeta, abstractmethod
import os

class DatasetBase(object):
    '''Dataset class.'''

    __metaclass__ = ABCMeta

    def __init__(self, datastore=None, verbose=False):
        self._verbose = verbose

        if datastore is None:
            self._datastore = os.path.join(os.environ['HOME'], '.bdset')
        else:
            self._datastore = datastore
        if self._verbose:
            print('Initialize dataset.')
            print('Data store: %s' % self._datastore)

    @abstractmethod
    def get(self, **kargs):
        pass
