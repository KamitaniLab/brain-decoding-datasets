'''Downloader.'''


__all__ = ['download_file']


import os
import urllib


def download_file(url, save_path):
    '''Download a file from `url`.'''

    print('Downloading %s' % url)
    urllib.urlretrieve(url, save_path)
    print('Saved %s' % save_path)
