# Brain decoding datasets

This Python package provides interfaces to datasets published from Kamitani Lab, Kyoto Univ and ATR.

## Installation

``` shellsession
$ pip install git+https://github.com/KamitaniLab/brain-decoding-datasets.git
```

## Quick guide

``` shellsession
import bdds

# Getting decoded DNN feeature datasets
dataset_dnn = bdds.DecodedDNN('data/decodeddnn')

decoded_features = dataset_dnn.get(mode='decoded', subject='S1', net='AlexNet', layer='fc8')
```

## Supported datasets

- [Generic object decoding](https://github.com/KamitaniLab/GenericObjectDecoding)
    - [Horikawa & Kamitani (2017) Generic decoding of seen and imagined objects using hierarchical visual features. Nat Commun.](https://www.nature.com/articles/ncomms15037)
- Decoded DNN
