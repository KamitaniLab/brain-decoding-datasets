'''Decoded DNN features.'''

__all__ = ['DecodedDNN']

import os
import warnings
from itertools import product
from collections import OrderedDict

import numpy as np
import scipy.io as sio
import h5py

from .bdds import DatasetBase


class DecodedDNN(DatasetBase):
    '''Decoded DNN feature dataset class.'''

    __subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'Average']
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

    def __init__(self, datastore=None, verbose=False):
        super(DecodedDNN, self).__init__(datastore=datastore, verbose=verbose)
        # Default data store path
        if datastore is None:
            self._datastore = os.path.join(self._datastore, 'decodeddnn')

    def get(self, mode=None, subject=None, net=None, layer=None, image=None, return_dict=False, force_list=False):
        '''Returns selected data.'''

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
        output = []
        
        for nt in net:
            lys = net_dict[nt]
            for sb, ly, im in product(subject, lys, image):
                fpath = self.__get_filepath(mode=mode, subject=sb, net=nt, layer=ly, image=im)
                if self._verbose: print('Data file: %s' % fpath)
                dat = self.__load_data(fpath)
                res = {'mode' : mode,
                       'subject' : sb,
                       'net' : nt,
                       'layer' : ly,
                       'data' : dat}
                if im is not None: res.update({'image' : im})
                output.append(res)

        # Postprocessings
        if not return_dict:
            # Returns array(s) instead of dict(s)
            output = [out['data'] for out in output]
        
        if not force_list:
            # Always returns a list
            output = output[0] if len(output) == 1 else output
                
        return output

    def __get_filepath(self, mode=None, subject=None, net=None, layer=None, image=None):
        if mode == 'decoded':
            fpath = os.path.join(mode, net, layer, subject, '%s_%s.mat' % (layer, image))
        elif mode == 'accuracy':
            fpath = os.path.join(mode, net, layer, '%s_%s_%s.mat' % (layer, mode, subject))
        elif mode == 'rank':
            fpath = os.path.join(mode, net, layer, '%s_%s_%s.mat' % (layer, mode, subject))
        else:
            raise ValueError('Unknown mode: %s' % mode)

        return fpath

    def __load_data(self, fpath):
        flocalpath = os.path.join(self._datastore, fpath)
        if not os.path.exists(flocalpath):
            if self._verbose: print('Downloading file %s...' % fpath)
            dlfile = self.__download_data('')

        if self._verbose: print('Loading %s...' % flocalpath)
        dat = self.__load_data_local(flocalpath)
            
        return dat

    def __load_data_local(self, fpath):
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

    def __download_data(self, url):
        raise RuntimeError('Data donwloading is not supported yet.')
        return np.empty([])
    
    def __listize(self, x):
        return x if isinstance(x, list) else [x]
