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

- Decoded DNN
