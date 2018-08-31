'''Downloading script.

Usage
-----

    $ python -m bddpy.downloadall handshape
'''


import os
import sys
from argparse import ArgumentParser

# bdds
from .decodeddnn import *   # Decoded DNN features from fMRI data
from .god import *          # Generic object decoding dataset
from .fmri import *         # fMRI datasets


# Entry point #################################################################

def main():
    # Command line arguments
    argparse = ArgumentParser(description=__doc__)
    argparse.add_argument('dataset', type=str, help='Dataset', default=None)
    argparse.add_argument('--output', type=str, help='Output directory', default=None)
    args = argparse.parse_args()

    # Dataset tables
    dataset_table = {'handshape': HandShapeDecoding,
                     'god': GenericObjectDecoding,
                     'decodeddnn': DecodedDNN}
    
    # Init dataset instrance
    ds = init_dataset(dataset_table[args.dataset], args.output)

    # Download all files
    ds.download_all()

    
def init_dataset(dataset, datastore):
    return dataset(datastore=datastore, auto_download=True, verbose=True)
    

if __name__ == '__main__':
    main()
