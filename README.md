# Brain decoding datasets

This Python package provides interfaces to datasets published from Kamitani Lab, Kyoto Univ and ATR.

## Installation

``` shellsession
$ pip install git+https://github.com/KamitaniLab/brain-decoding-datasets.git
```

## Quick guide

``` shellsession
import bdds


# GOD fMRI dataset

dataset = bdds.GenericObjectDecoding('data/god')

data_s1 = dataset.get(mode='fmri', subject='Subject1')  # Return fMRI data as a bdpy dataset


# Decoded DNN features

dataset_dnn = bdds.DecodedDNN('data/decodeddnn')

feature = dataset_dnn.get(mode='decoded', subject='S1', net='AlexNet', layer='fc8')  # Return features as a numpy array
```

## Supported datasets

- [Generic object decoding](https://github.com/KamitaniLab/GenericObjectDecoding)
    - [Horikawa & Kamitani (2017) Generic decoding of seen and imagined objects using hierarchical visual features. Nat Commun.](https://www.nature.com/articles/ncomms15037)
- Decoded DNN
