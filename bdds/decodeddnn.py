'''Decoded DNN features.'''

__all__ = ['DecodedDNN']

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

from .bdds import DatasetBase


class DecodedDNN(DatasetBase):
    '''Decoded DNN feature dataset class.

    Attributes
    ----------
    datastore : str, optional
        Path to data store directory
    verbose : bool, optional
        Output verbose messages or not
    '''

    __subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'Averaged']
    __nets = {'AlexNet' : ['conv1', 'conv2', 'conv3', 'conv4', 'conv5',
                           'fc6', 'fc7', 'fc8',
                           'norm1', 'norm2',
                           'pool1', 'pool2', 'pool5',
                           'prob',
                           'relu1', 'relu2', 'relu3', 'relu4', 'relu5', 'relu6', 'relu7'],
              'VGG19' : ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4',
                         'drop6', 'drop7',
                         'fc6', 'fc7', 'fc8',
                         'pool1', 'pool2', 'pool3', 'pool4', 'pool5',
                         'prob',
                         'relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2', 'relu3_3', 'relu3_4', 'relu4_1', 'relu4_2', 'relu4_3', 'relu4_4', 'relu5_1', 'relu5_2', 'relu5_3', 'relu5_4', 'relu6', 'relu7']}
    __images = ['n01443537_22563', 'n01621127_19020', 'n01677366_18182', 'n01846331_17038', 'n01858441_11077',
                'n01943899_24131', 'n01976957_13223', 'n02071294_46212', 'n02128385_20264', 'n02139199_10398',
                'n02190790_15121', 'n02274259_24319', 'n02416519_12793', 'n02437136_12836', 'n02437971_5013',
                'n02690373_7713', 'n02797295_15411', 'n02824058_18729', 'n02882301_14188', 'n02916179_24850',
                'n02950256_22949', 'n02951358_23759', 'n03064758_38750', 'n03122295_31279', 'n03124170_13920',
                'n03237416_58334', 'n03272010_11001', 'n03345837_12501', 'n03379051_8496', 'n03452741_24622',
                'n03455488_28622', 'n03482252_22530', 'n03495258_9895', 'n03584254_5040', 'n03626115_19498',
                'n03710193_22225', 'n03716966_28524', 'n03761084_43533', 'n03767745_109', 'n03941684_21672',
                'n03954393_10038', 'n04210120_9062', 'n04252077_10859', 'n04254777_16338', 'n04297750_25624',
                'n04387400_16693', 'n04507155_21299', 'n04533802_19479', 'n04554684_53399', 'n04572121_3262']
    __modes = ['decoded', 'accuracy', 'rank']

    __remote_files = {'decodedDNN-accuracy-AlexNet-conv1.zip': 'https://ndownloader.figshare.com/files/11877389',
                      'decodedDNN-accuracy-AlexNet-conv2.zip': 'https://ndownloader.figshare.com/files/11877392',
                      'decodedDNN-accuracy-AlexNet-conv3.zip': 'https://ndownloader.figshare.com/files/11877398',
                      'decodedDNN-accuracy-AlexNet-conv4.zip': 'https://ndownloader.figshare.com/files/11877401',
                      'decodedDNN-accuracy-AlexNet-conv5.zip': 'https://ndownloader.figshare.com/files/11877407',
                      'decodedDNN-accuracy-AlexNet-fc6.zip': 'https://ndownloader.figshare.com/files/11877410',
                      'decodedDNN-accuracy-AlexNet-fc7.zip': 'https://ndownloader.figshare.com/files/11877416',
                      'decodedDNN-accuracy-AlexNet-fc8.zip': 'https://ndownloader.figshare.com/files/11877422',
                      'decodedDNN-accuracy-AlexNet-norm1.zip': 'https://ndownloader.figshare.com/files/11877434',
                      'decodedDNN-accuracy-AlexNet-norm2.zip': 'https://ndownloader.figshare.com/files/11877437',
                      'decodedDNN-accuracy-AlexNet-pool1.zip': 'https://ndownloader.figshare.com/files/11877443',
                      'decodedDNN-accuracy-AlexNet-pool2.zip': 'https://ndownloader.figshare.com/files/11877446',
                      'decodedDNN-accuracy-AlexNet-pool5.zip': 'https://ndownloader.figshare.com/files/11877449',
                      'decodedDNN-accuracy-AlexNet-prob.zip': 'https://ndownloader.figshare.com/files/11877452',
                      'decodedDNN-accuracy-AlexNet-relu1.zip': 'https://ndownloader.figshare.com/files/11877455',
                      'decodedDNN-accuracy-AlexNet-relu2.zip': 'https://ndownloader.figshare.com/files/11877458',
                      'decodedDNN-accuracy-AlexNet-relu3.zip': 'https://ndownloader.figshare.com/files/11877461',
                      'decodedDNN-accuracy-AlexNet-relu4.zip': 'https://ndownloader.figshare.com/files/11877464',
                      'decodedDNN-accuracy-AlexNet-relu5.zip': 'https://ndownloader.figshare.com/files/11877467',
                      'decodedDNN-accuracy-AlexNet-relu6.zip': 'https://ndownloader.figshare.com/files/11877470',
                      'decodedDNN-accuracy-AlexNet-relu7.zip': 'https://ndownloader.figshare.com/files/11877473',
                      'decodedDNN-decoded-AlexNet-conv1.zip': 'https://ndownloader.figshare.com/files/11877479',
                      'decodedDNN-decoded-AlexNet-conv2.zip': 'https://ndownloader.figshare.com/files/11877485',
                      'decodedDNN-decoded-AlexNet-conv3.zip': 'https://ndownloader.figshare.com/files/11877488',
                      'decodedDNN-decoded-AlexNet-conv4.zip': 'https://ndownloader.figshare.com/files/11877491',
                      'decodedDNN-decoded-AlexNet-conv5.zip': 'https://ndownloader.figshare.com/files/11877494',
                      'decodedDNN-decoded-AlexNet-fc6.zip': 'https://ndownloader.figshare.com/files/11877497',
                      'decodedDNN-decoded-AlexNet-fc7.zip': 'https://ndownloader.figshare.com/files/11877500',
                      'decodedDNN-decoded-AlexNet-fc8.zip': 'https://ndownloader.figshare.com/files/11877503',
                      'decodedDNN-decoded-AlexNet-norm1.zip': 'https://ndownloader.figshare.com/files/11877509',
                      'decodedDNN-decoded-AlexNet-norm2.zip': 'https://ndownloader.figshare.com/files/11877512',
                      'decodedDNN-decoded-AlexNet-pool1.zip': 'https://ndownloader.figshare.com/files/11877518',
                      'decodedDNN-decoded-AlexNet-pool2.zip': 'https://ndownloader.figshare.com/files/11877521',
                      'decodedDNN-decoded-AlexNet-pool5.zip': 'https://ndownloader.figshare.com/files/11877527',
                      'decodedDNN-decoded-AlexNet-prob.zip': 'https://ndownloader.figshare.com/files/11877524',
                      'decodedDNN-decoded-AlexNet-relu1.zip': 'https://ndownloader.figshare.com/files/11877533',
                      'decodedDNN-decoded-AlexNet-relu2.zip': 'https://ndownloader.figshare.com/files/11877536',
                      'decodedDNN-decoded-AlexNet-relu3.zip': 'https://ndownloader.figshare.com/files/11877539',
                      'decodedDNN-decoded-AlexNet-relu4.zip': 'https://ndownloader.figshare.com/files/11877542',
                      'decodedDNN-decoded-AlexNet-relu5.zip': 'https://ndownloader.figshare.com/files/11877545',
                      'decodedDNN-decoded-AlexNet-relu6.zip': 'https://ndownloader.figshare.com/files/11877548',
                      'decodedDNN-decoded-AlexNet-relu7.zip': 'https://ndownloader.figshare.com/files/11877551',
                      'decodedDNN-rank-AlexNet-conv1.zip': 'https://ndownloader.figshare.com/files/11877554',
                      'decodedDNN-rank-AlexNet-conv2.zip': 'https://ndownloader.figshare.com/files/11877557',
                      'decodedDNN-rank-AlexNet-conv3.zip': 'https://ndownloader.figshare.com/files/11877560',
                      'decodedDNN-rank-AlexNet-conv4.zip': 'https://ndownloader.figshare.com/files/11877563',
                      'decodedDNN-rank-AlexNet-conv5.zip': 'https://ndownloader.figshare.com/files/11877566',
                      'decodedDNN-rank-AlexNet-fc6.zip': 'https://ndownloader.figshare.com/files/11877569',
                      'decodedDNN-rank-AlexNet-fc7.zip': 'https://ndownloader.figshare.com/files/11877572',
                      'decodedDNN-rank-AlexNet-fc8.zip': 'https://ndownloader.figshare.com/files/11877575',
                      'decodedDNN-rank-AlexNet-norm1.zip': 'https://ndownloader.figshare.com/files/11877578',
                      'decodedDNN-rank-AlexNet-norm2.zip': 'https://ndownloader.figshare.com/files/11877581',
                      'decodedDNN-rank-AlexNet-pool1.zip': 'https://ndownloader.figshare.com/files/11877584',
                      'decodedDNN-rank-AlexNet-pool2.zip': 'https://ndownloader.figshare.com/files/11877587',
                      'decodedDNN-rank-AlexNet-pool5.zip': 'https://ndownloader.figshare.com/files/11877590',
                      'decodedDNN-rank-AlexNet-prob.zip': 'https://ndownloader.figshare.com/files/11877593',
                      'decodedDNN-rank-AlexNet-relu1.zip': 'https://ndownloader.figshare.com/files/11877596',
                      'decodedDNN-rank-AlexNet-relu2.zip': 'https://ndownloader.figshare.com/files/11877599',
                      'decodedDNN-rank-AlexNet-relu3.zip': 'https://ndownloader.figshare.com/files/11877602',
                      'decodedDNN-rank-AlexNet-relu4.zip': 'https://ndownloader.figshare.com/files/11877605',
                      'decodedDNN-rank-AlexNet-relu5.zip': 'https://ndownloader.figshare.com/files/11877608',
                      'decodedDNN-rank-AlexNet-relu6.zip': 'https://ndownloader.figshare.com/files/11877611',
                      'decodedDNN-rank-AlexNet-relu7.zip': 'https://ndownloader.figshare.com/files/11877614',
                      'decodedDNN-true-AlexNet-conv1.zip': 'https://ndownloader.figshare.com/files/11877620',
                      'decodedDNN-true-AlexNet-conv2.zip': 'https://ndownloader.figshare.com/files/11877623',
                      'decodedDNN-true-AlexNet-conv3.zip': 'https://ndownloader.figshare.com/files/11877626',
                      'decodedDNN-true-AlexNet-conv4.zip': 'https://ndownloader.figshare.com/files/11877629',
                      'decodedDNN-true-AlexNet-conv5.zip': 'https://ndownloader.figshare.com/files/11877632',
                      'decodedDNN-true-AlexNet-fc6.zip': 'https://ndownloader.figshare.com/files/11877638',
                      'decodedDNN-true-AlexNet-fc7.zip': 'https://ndownloader.figshare.com/files/11877635',
                      'decodedDNN-true-AlexNet-fc8.zip': 'https://ndownloader.figshare.com/files/11877641',
                      'decodedDNN-true-AlexNet-norm1.zip': 'https://ndownloader.figshare.com/files/11877644',
                      'decodedDNN-true-AlexNet-norm2.zip': 'https://ndownloader.figshare.com/files/11877647',
                      'decodedDNN-true-AlexNet-pool1.zip': 'https://ndownloader.figshare.com/files/11877650',
                      'decodedDNN-true-AlexNet-pool2.zip': 'https://ndownloader.figshare.com/files/11877653',
                      'decodedDNN-true-AlexNet-pool5.zip': 'https://ndownloader.figshare.com/files/11877656',
                      'decodedDNN-true-AlexNet-prob.zip': 'https://ndownloader.figshare.com/files/11877659',
                      'decodedDNN-true-AlexNet-relu1.zip': 'https://ndownloader.figshare.com/files/11877662',
                      'decodedDNN-true-AlexNet-relu2.zip': 'https://ndownloader.figshare.com/files/11877665',
                      'decodedDNN-true-AlexNet-relu3.zip': 'https://ndownloader.figshare.com/files/11877668',
                      'decodedDNN-true-AlexNet-relu4.zip': 'https://ndownloader.figshare.com/files/11877671',
                      'decodedDNN-true-AlexNet-relu5.zip': 'https://ndownloader.figshare.com/files/11877674',
                      'decodedDNN-true-AlexNet-relu6.zip': 'https://ndownloader.figshare.com/files/11877677',
                      'decodedDNN-true-AlexNet-relu7.zip': 'https://ndownloader.figshare.com/files/11877680',
                      'decodedDNN-accuracy-VGG19-conv1_1.zip': 'https://ndownloader.figshare.com/files/11877764',
                      'decodedDNN-accuracy-VGG19-conv1_2.zip': 'https://ndownloader.figshare.com/files/11877767',
                      'decodedDNN-accuracy-VGG19-conv2_1.zip': 'https://ndownloader.figshare.com/files/11877770',
                      'decodedDNN-accuracy-VGG19-conv2_2.zip': 'https://ndownloader.figshare.com/files/11877773',
                      'decodedDNN-accuracy-VGG19-conv3_1.zip': 'https://ndownloader.figshare.com/files/11877776',
                      'decodedDNN-accuracy-VGG19-conv3_2.zip': 'https://ndownloader.figshare.com/files/11877779',
                      'decodedDNN-accuracy-VGG19-conv3_3.zip': 'https://ndownloader.figshare.com/files/11877782',
                      'decodedDNN-accuracy-VGG19-conv3_4.zip': 'https://ndownloader.figshare.com/files/11877785',
                      'decodedDNN-accuracy-VGG19-conv4_1.zip': 'https://ndownloader.figshare.com/files/11877788',
                      'decodedDNN-accuracy-VGG19-conv4_2.zip': 'https://ndownloader.figshare.com/files/11877791',
                      'decodedDNN-accuracy-VGG19-conv4_3.zip': 'https://ndownloader.figshare.com/files/11877794',
                      'decodedDNN-accuracy-VGG19-conv4_4.zip': 'https://ndownloader.figshare.com/files/11877797',
                      'decodedDNN-accuracy-VGG19-conv5_1.zip': 'https://ndownloader.figshare.com/files/11877800',
                      'decodedDNN-accuracy-VGG19-conv5_2.zip': 'https://ndownloader.figshare.com/files/11877803',
                      'decodedDNN-accuracy-VGG19-conv5_3.zip': 'https://ndownloader.figshare.com/files/11877806',
                      'decodedDNN-accuracy-VGG19-conv5_4.zip': 'https://ndownloader.figshare.com/files/11877809',
                      'decodedDNN-accuracy-VGG19-drop6.zip': 'https://ndownloader.figshare.com/files/11877812',
                      'decodedDNN-accuracy-VGG19-drop7.zip': 'https://ndownloader.figshare.com/files/11877815',
                      'decodedDNN-accuracy-VGG19-fc6.zip': 'https://ndownloader.figshare.com/files/11877818',
                      'decodedDNN-accuracy-VGG19-fc7.zip': 'https://ndownloader.figshare.com/files/11877821',
                      'decodedDNN-accuracy-VGG19-fc8.zip': 'https://ndownloader.figshare.com/files/11877824',
                      'decodedDNN-accuracy-VGG19-pool1.zip': 'https://ndownloader.figshare.com/files/11877827',
                      'decodedDNN-accuracy-VGG19-pool2.zip': 'https://ndownloader.figshare.com/files/11877830',
                      'decodedDNN-accuracy-VGG19-pool3.zip': 'https://ndownloader.figshare.com/files/11877833',
                      'decodedDNN-accuracy-VGG19-pool4.zip': 'https://ndownloader.figshare.com/files/11877836',
                      'decodedDNN-accuracy-VGG19-pool5.zip': 'https://ndownloader.figshare.com/files/11877839',
                      'decodedDNN-accuracy-VGG19-prob.zip': 'https://ndownloader.figshare.com/files/11877842',
                      'decodedDNN-accuracy-VGG19-relu1_1.zip': 'https://ndownloader.figshare.com/files/11877845',
                      'decodedDNN-accuracy-VGG19-relu1_2.zip': 'https://ndownloader.figshare.com/files/11877848',
                      'decodedDNN-accuracy-VGG19-relu2_1.zip': 'https://ndownloader.figshare.com/files/11877851',
                      'decodedDNN-accuracy-VGG19-relu2_2.zip': 'https://ndownloader.figshare.com/files/11877854',
                      'decodedDNN-accuracy-VGG19-relu3_1.zip': 'https://ndownloader.figshare.com/files/11877857',
                      'decodedDNN-accuracy-VGG19-relu3_2.zip': 'https://ndownloader.figshare.com/files/11877863',
                      'decodedDNN-accuracy-VGG19-relu3_3.zip': 'https://ndownloader.figshare.com/files/11877866',
                      'decodedDNN-accuracy-VGG19-relu3_4.zip': 'https://ndownloader.figshare.com/files/11877869',
                      'decodedDNN-accuracy-VGG19-relu4_1.zip': 'https://ndownloader.figshare.com/files/11877872',
                      'decodedDNN-accuracy-VGG19-relu4_2.zip': 'https://ndownloader.figshare.com/files/11877875',
                      'decodedDNN-accuracy-VGG19-relu4_3.zip': 'https://ndownloader.figshare.com/files/11877878',
                      'decodedDNN-accuracy-VGG19-relu4_4.zip': 'https://ndownloader.figshare.com/files/11877881',
                      'decodedDNN-accuracy-VGG19-relu5_1.zip': 'https://ndownloader.figshare.com/files/11877884',
                      'decodedDNN-accuracy-VGG19-relu5_2.zip': 'https://ndownloader.figshare.com/files/11877887',
                      'decodedDNN-accuracy-VGG19-relu5_3.zip': 'https://ndownloader.figshare.com/files/11877890',
                      'decodedDNN-accuracy-VGG19-relu5_4.zip': 'https://ndownloader.figshare.com/files/11877893',
                      'decodedDNN-accuracy-VGG19-relu6.zip': 'https://ndownloader.figshare.com/files/11877896',
                      'decodedDNN-accuracy-VGG19-relu7.zip': 'https://ndownloader.figshare.com/files/11877899',
                      'decodedDNN-rank-VGG19-conv1_1.zip': 'https://ndownloader.figshare.com/files/11877902',
                      'decodedDNN-rank-VGG19-conv1_2.zip': 'https://ndownloader.figshare.com/files/11877905',
                      'decodedDNN-rank-VGG19-conv2_1.zip': 'https://ndownloader.figshare.com/files/11877908',
                      'decodedDNN-rank-VGG19-conv2_2.zip': 'https://ndownloader.figshare.com/files/11877911',
                      'decodedDNN-rank-VGG19-conv3_1.zip': 'https://ndownloader.figshare.com/files/11877914',
                      'decodedDNN-rank-VGG19-conv3_2.zip': 'https://ndownloader.figshare.com/files/11877917',
                      'decodedDNN-rank-VGG19-conv3_3.zip': 'https://ndownloader.figshare.com/files/11877920',
                      'decodedDNN-rank-VGG19-conv3_4.zip': 'https://ndownloader.figshare.com/files/11877923',
                      'decodedDNN-rank-VGG19-conv4_1.zip': 'https://ndownloader.figshare.com/files/11877926',
                      'decodedDNN-rank-VGG19-conv4_2.zip': 'https://ndownloader.figshare.com/files/11877929',
                      'decodedDNN-rank-VGG19-conv4_3.zip': 'https://ndownloader.figshare.com/files/11877932',
                      'decodedDNN-rank-VGG19-conv4_4.zip': 'https://ndownloader.figshare.com/files/11877935',
                      'decodedDNN-rank-VGG19-conv5_1.zip': 'https://ndownloader.figshare.com/files/11877941',
                      'decodedDNN-rank-VGG19-conv5_2.zip': 'https://ndownloader.figshare.com/files/11877938',
                      'decodedDNN-rank-VGG19-conv5_3.zip': 'https://ndownloader.figshare.com/files/11877944',
                      'decodedDNN-rank-VGG19-conv5_4.zip': 'https://ndownloader.figshare.com/files/11877947',
                      'decodedDNN-rank-VGG19-drop6.zip': 'https://ndownloader.figshare.com/files/11877950',
                      'decodedDNN-rank-VGG19-drop7.zip': 'https://ndownloader.figshare.com/files/11877953',
                      'decodedDNN-rank-VGG19-fc6.zip': 'https://ndownloader.figshare.com/files/11877956',
                      'decodedDNN-rank-VGG19-fc7.zip': 'https://ndownloader.figshare.com/files/11877959',
                      'decodedDNN-rank-VGG19-fc8.zip': 'https://ndownloader.figshare.com/files/11877962',
                      'decodedDNN-rank-VGG19-pool1.zip': 'https://ndownloader.figshare.com/files/11877965',
                      'decodedDNN-rank-VGG19-pool2.zip': 'https://ndownloader.figshare.com/files/11877968',
                      'decodedDNN-rank-VGG19-pool3.zip': 'https://ndownloader.figshare.com/files/11877971',
                      'decodedDNN-rank-VGG19-pool4.zip': 'https://ndownloader.figshare.com/files/11877974',
                      'decodedDNN-rank-VGG19-pool5.zip': 'https://ndownloader.figshare.com/files/11877977',
                      'decodedDNN-rank-VGG19-prob.zip': 'https://ndownloader.figshare.com/files/11877980',
                      'decodedDNN-rank-VGG19-relu1_1.zip': 'https://ndownloader.figshare.com/files/11877983',
                      'decodedDNN-rank-VGG19-relu1_2.zip': 'https://ndownloader.figshare.com/files/11877986',
                      'decodedDNN-rank-VGG19-relu2_1.zip': 'https://ndownloader.figshare.com/files/11877989',
                      'decodedDNN-rank-VGG19-relu2_2.zip': 'https://ndownloader.figshare.com/files/11877992',
                      'decodedDNN-rank-VGG19-relu3_1.zip': 'https://ndownloader.figshare.com/files/11877995',
                      'decodedDNN-rank-VGG19-relu3_2.zip': 'https://ndownloader.figshare.com/files/11877998',
                      'decodedDNN-rank-VGG19-relu3_3.zip': 'https://ndownloader.figshare.com/files/11878001',
                      'decodedDNN-rank-VGG19-relu3_4.zip': 'https://ndownloader.figshare.com/files/11878004',
                      'decodedDNN-rank-VGG19-relu4_1.zip': 'https://ndownloader.figshare.com/files/11878007',
                      'decodedDNN-rank-VGG19-relu4_2.zip': 'https://ndownloader.figshare.com/files/11878010',
                      'decodedDNN-rank-VGG19-relu4_3.zip': 'https://ndownloader.figshare.com/files/11878013',
                      'decodedDNN-rank-VGG19-relu4_4.zip': 'https://ndownloader.figshare.com/files/11878016',
                      'decodedDNN-rank-VGG19-relu5_1.zip': 'https://ndownloader.figshare.com/files/11878019',
                      'decodedDNN-rank-VGG19-relu5_2.zip': 'https://ndownloader.figshare.com/files/11878022',
                      'decodedDNN-rank-VGG19-relu5_3.zip': 'https://ndownloader.figshare.com/files/11878025',
                      'decodedDNN-rank-VGG19-relu5_4.zip': 'https://ndownloader.figshare.com/files/11878028',
                      'decodedDNN-rank-VGG19-relu6.zip': 'https://ndownloader.figshare.com/files/11878031',
                      'decodedDNN-rank-VGG19-relu7.zip': 'https://ndownloader.figshare.com/files/11878034',
                      'decodedDNN-true-VGG19-conv1_1.zip': 'https://ndownloader.figshare.com/files/11878037',
                      'decodedDNN-true-VGG19-conv1_2.zip': 'https://ndownloader.figshare.com/files/11878088',
                      'decodedDNN-true-VGG19-conv2_1.zip': 'https://ndownloader.figshare.com/files/11878097',
                      'decodedDNN-true-VGG19-conv2_2.zip': 'https://ndownloader.figshare.com/files/11878100',
                      'decodedDNN-true-VGG19-conv3_1.zip': 'https://ndownloader.figshare.com/files/11878103',
                      'decodedDNN-true-VGG19-conv3_2.zip': 'https://ndownloader.figshare.com/files/11878106',
                      'decodedDNN-true-VGG19-conv3_3.zip': 'https://ndownloader.figshare.com/files/11878109',
                      'decodedDNN-true-VGG19-conv3_4.zip': 'https://ndownloader.figshare.com/files/11878112',
                      'decodedDNN-true-VGG19-conv4_1.zip': 'https://ndownloader.figshare.com/files/11878115',
                      'decodedDNN-true-VGG19-conv4_2.zip': 'https://ndownloader.figshare.com/files/11878118',
                      'decodedDNN-true-VGG19-conv4_3.zip': 'https://ndownloader.figshare.com/files/11878121',
                      'decodedDNN-true-VGG19-conv4_4.zip': 'https://ndownloader.figshare.com/files/11878124',
                      'decodedDNN-true-VGG19-conv5_1.zip': 'https://ndownloader.figshare.com/files/11878127',
                      'decodedDNN-true-VGG19-conv5_2.zip': 'https://ndownloader.figshare.com/files/11878130',
                      'decodedDNN-true-VGG19-conv5_3.zip': 'https://ndownloader.figshare.com/files/11878133',
                      'decodedDNN-true-VGG19-conv5_4.zip': 'https://ndownloader.figshare.com/files/11878136',
                      'decodedDNN-true-VGG19-drop6.zip': 'https://ndownloader.figshare.com/files/11878139',
                      'decodedDNN-true-VGG19-drop7.zip': 'https://ndownloader.figshare.com/files/11878142',
                      'decodedDNN-true-VGG19-fc6.zip': 'https://ndownloader.figshare.com/files/11878148',
                      'decodedDNN-true-VGG19-fc7.zip': 'https://ndownloader.figshare.com/files/11878145',
                      'decodedDNN-true-VGG19-fc8.zip': 'https://ndownloader.figshare.com/files/11878151',
                      'decodedDNN-true-VGG19-pool1.zip': 'https://ndownloader.figshare.com/files/11878154',
                      'decodedDNN-true-VGG19-pool2.zip': 'https://ndownloader.figshare.com/files/11878157',
                      'decodedDNN-true-VGG19-pool3.zip': 'https://ndownloader.figshare.com/files/11878160',
                      'decodedDNN-true-VGG19-pool4.zip': 'https://ndownloader.figshare.com/files/11878163',
                      'decodedDNN-true-VGG19-pool5.zip': 'https://ndownloader.figshare.com/files/11878166',
                      'decodedDNN-true-VGG19-prob.zip': 'https://ndownloader.figshare.com/files/11878169',
                      'decodedDNN-true-VGG19-relu1_1.zip': 'https://ndownloader.figshare.com/files/11878172',
                      'decodedDNN-true-VGG19-relu1_2.zip': 'https://ndownloader.figshare.com/files/11878175',
                      'decodedDNN-true-VGG19-relu2_1.zip': 'https://ndownloader.figshare.com/files/11878178',
                      'decodedDNN-true-VGG19-relu2_2.zip': 'https://ndownloader.figshare.com/files/11878181',
                      'decodedDNN-true-VGG19-relu3_1.zip': 'https://ndownloader.figshare.com/files/11878184',
                      'decodedDNN-true-VGG19-relu3_2.zip': 'https://ndownloader.figshare.com/files/11878187',
                      'decodedDNN-true-VGG19-relu3_3.zip': 'https://ndownloader.figshare.com/files/11878190',
                      'decodedDNN-true-VGG19-relu3_4.zip': 'https://ndownloader.figshare.com/files/11878196',
                      'decodedDNN-true-VGG19-relu4_1.zip': 'https://ndownloader.figshare.com/files/11878199',
                      'decodedDNN-true-VGG19-relu4_2.zip': 'https://ndownloader.figshare.com/files/11878202',
                      'decodedDNN-true-VGG19-relu4_3.zip': 'https://ndownloader.figshare.com/files/11878205',
                      'decodedDNN-true-VGG19-relu4_4.zip': 'https://ndownloader.figshare.com/files/11878208',
                      'decodedDNN-true-VGG19-relu5_1.zip': 'https://ndownloader.figshare.com/files/11878214',
                      'decodedDNN-true-VGG19-relu5_2.zip': 'https://ndownloader.figshare.com/files/11878211',
                      'decodedDNN-true-VGG19-relu5_3.zip': 'https://ndownloader.figshare.com/files/11878217',
                      'decodedDNN-true-VGG19-relu5_4.zip': 'https://ndownloader.figshare.com/files/11878220',
                      'decodedDNN-true-VGG19-relu6.zip': 'https://ndownloader.figshare.com/files/11878223',
                      'decodedDNN-true-VGG19-relu7.zip': 'https://ndownloader.figshare.com/files/11878226',
                      'decodedDNN-decoded-VGG19-conv1_1.zip': 'https://ndownloader.figshare.com/files/11878313',
                      'decodedDNN-decoded-VGG19-conv1_2.zip': 'https://ndownloader.figshare.com/files/11878361',
                      'decodedDNN-decoded-VGG19-conv2_1.zip': 'https://ndownloader.figshare.com/files/11878367',
                      'decodedDNN-decoded-VGG19-conv2_2.zip': 'https://ndownloader.figshare.com/files/11878385',
                      'decodedDNN-decoded-VGG19-conv3_1.zip': 'https://ndownloader.figshare.com/files/11878421',
                      'decodedDNN-decoded-VGG19-conv3_2.zip': 'https://ndownloader.figshare.com/files/11878430',
                      'decodedDNN-decoded-VGG19-conv3_3.zip': 'https://ndownloader.figshare.com/files/11878439',
                      'decodedDNN-decoded-VGG19-conv3_4.zip': 'https://ndownloader.figshare.com/files/11878445'}

    def __init__(self, datastore=None, verbose=False):
        super(DecodedDNN, self).__init__(datastore=datastore, verbose=verbose)
        # Default data store path
        if datastore is None:
            self._datastore = os.path.join(self._datastore, 'decodeddnn')

    def _get_files(self, mode=None, subject=None, net=None, layer=None, image=None):

        # Input processing
        if mode is None: raise RuntimeError('`mode` is required.')

        if subject is None: subject = DecodedDNN.__subjects
        subject = self.__listize(subject)

        if net is None: net = list(DecodedDNN.__nets.keys())
        net = self.__listize(net)

        net_dict = {}
        for nt in net:
            layer_tmp = DecodedDNN.__nets[nt] if layer is None else layer
            layer_tmp = self.__listize(layer_tmp)
            net_dict.update({nt : layer_tmp})

        if mode == 'decoded':
            if image is None: image = DecodedDNN.__images
            image = self.__listize(image)
        else:
            if image is not None: warnings.warn('`image` is only effective for `decoded` mode.')
            image = [None]

        # Disp info
        if self._verbose:
            print('Mode :        %s' % mode)
            print('Subject :     %s' % subject)
            print('Net :         %s' % net)
            for n in net_dict: print('Layers (%s) : %s' % (n, net_dict[n]))
            if image is not None: print('Image :       %s' % image)

        # TODO: add args value check

        # Get data files
        collection = []

        for nt in net:
            lys = net_dict[nt]
            for sb, ly, im in product(subject, lys, image):
                fpath = self.__get_filepath(mode=mode, subject=sb, net=nt, layer=ly, image=im)
                if self._verbose: print('Data file: %s' % fpath)
                collection.append({'identifier': {'mode': mode,
                                                  'subject': sb,
                                                  'net': nt,
                                                  'layer': ly,
                                                  'image': im},
                                   'file': fpath,
                                   'data': None})

        return collection

    def __get_filepath(self, mode=None, subject=None, net=None, layer=None, image=None):
        if mode == 'decoded':
            fpath = os.path.join(mode, net, layer, subject, '%s-%s-%s-%s-%s.mat' % (mode, net, layer, subject, image))
        elif mode == 'accuracy':
            fpath = os.path.join(mode, net, layer, '%s-%s-%s-%s.mat' % (mode, net, layer, subject))
        elif mode == 'rank':
            fpath = os.path.join(mode, net, layer, '%s-%s-%s-%s.mat' % (mode, net, layer, subject))
        else:
            raise ValueError('Unknown mode: %s' % mode)

        return fpath

    def _load_file(self, fpath):
        try:
            d = sio.loadmat(fpath)
            if 'feat' in d: return d['feat']
            if 'accuracy' in d: return d['accuracy']
            if 'rank' in d: return d['rank']
        except:
            d = h5py.File(fpath)
            if 'feat' in d: return d['feat'].value
            if 'accuracy' in d: return d['accuracy'].value
            if 'rank' in d: return d['rank'].value

        raise RuntimeError('Invalid data: %s' % fpath)

    def _download_file(self, fname):
        fstr = fname.split('/')
        remote_file = 'decodedDNN-%s-%s-%s.zip' % (fstr[0], fstr[1], fstr[2])
        url = DecodedDNN.__remote_files[remote_file]

        private_link = '76f1e5e10fdfacc5ec12'
        url = '%s?private_link=%s' % (url, private_link)

        t = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        tempdir = os.path.join(self._datastore, '.temp-%s' % t)
        tempfile = os.path.join(self._datastore, '.temp-%s.zip' % t)

        # Download file
        print('Downloading from %s' % url)
        urllib.urlretrieve(url, tempfile)

        # Unzip the file
        with zipfile.ZipFile(tempfile, 'r') as f:
            f.extractall(tempdir)

        # Move files
        for ds, ss, fs in os.walk(tempdir):
            if not fs: continue
            for f in fs:
                src = os.path.join(ds, f)
                trg = os.path.join('data/decodeddnn',
                                   src.split('/decodedDNN/')[1])
                trg_dir = os.path.dirname(trg)
                if not os.path.exists(trg_dir): os.makedirs(trg_dir)
                shutil.move(src, trg)

        # Remove temp files
        os.remove(tempfile)
        shutil.rmtree(tempdir)

        return np.empty([])

    def __listize(self, x):
        return x if isinstance(x, list) else [x]
