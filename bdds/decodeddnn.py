'''Decoded DNN features.'''

__all__ = ['DecodedDNN']

import os
import warnings
from itertools import product
from collections import OrderedDict
import zipfile
import shutil
import datetime

import numpy as np
import scipy.io as sio
import h5py

from .bdds import DatasetBase
from .download import download_file


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
                           'fc6', 'fc7', 'fc8'],
              'VGG19' : ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4',
                         'pool1', 'pool2', 'pool3', 'pool4', 'pool5']}
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
    __modes = ['decoded', 'accuracy', 'rank', 'true']

    __remote_files = {'decodedDNN-accuracy-AlexNet-conv1.zip': 'https://ndownloader.figshare.com/files/13077233',
                      'decodedDNN-accuracy-AlexNet-conv2.zip': 'https://ndownloader.figshare.com/files/13077236',
                      'decodedDNN-accuracy-AlexNet-conv3.zip': 'https://ndownloader.figshare.com/files/13077239',
                      'decodedDNN-accuracy-AlexNet-conv4.zip': 'https://ndownloader.figshare.com/files/13077242',
                      'decodedDNN-accuracy-AlexNet-conv5.zip': 'https://ndownloader.figshare.com/files/13077245',
                      'decodedDNN-accuracy-AlexNet-fc6.zip': 'https://ndownloader.figshare.com/files/13077248',
                      'decodedDNN-accuracy-AlexNet-fc7.zip': 'https://ndownloader.figshare.com/files/13077251',
                      'decodedDNN-accuracy-AlexNet-fc8.zip': 'https://ndownloader.figshare.com/files/13077254',
                      'decodedDNN-accuracy-VGG19-conv1_1.zip': 'https://ndownloader.figshare.com/files/13077314',
                      'decodedDNN-accuracy-VGG19-conv1_2.zip': 'https://ndownloader.figshare.com/files/13077317',
                      'decodedDNN-accuracy-VGG19-conv2_1.zip': 'https://ndownloader.figshare.com/files/13077320',
                      'decodedDNN-accuracy-VGG19-conv2_2.zip': 'https://ndownloader.figshare.com/files/13077323',
                      'decodedDNN-accuracy-VGG19-conv3_1.zip': 'https://ndownloader.figshare.com/files/13077326',
                      'decodedDNN-accuracy-VGG19-conv3_2.zip': 'https://ndownloader.figshare.com/files/13077329',
                      'decodedDNN-accuracy-VGG19-conv3_3.zip': 'https://ndownloader.figshare.com/files/13077332',
                      'decodedDNN-accuracy-VGG19-conv3_4.zip': 'https://ndownloader.figshare.com/files/13077335',
                      'decodedDNN-accuracy-VGG19-conv4_1.zip': 'https://ndownloader.figshare.com/files/13077338',
                      'decodedDNN-accuracy-VGG19-conv4_2.zip': 'https://ndownloader.figshare.com/files/13077284',
                      'decodedDNN-accuracy-VGG19-conv4_3.zip': 'https://ndownloader.figshare.com/files/13077287',
                      'decodedDNN-accuracy-VGG19-conv4_4.zip': 'https://ndownloader.figshare.com/files/13077290',
                      'decodedDNN-accuracy-VGG19-conv5_1.zip': 'https://ndownloader.figshare.com/files/13077293',
                      'decodedDNN-accuracy-VGG19-conv5_2.zip': 'https://ndownloader.figshare.com/files/13077296',
                      'decodedDNN-accuracy-VGG19-conv5_3.zip': 'https://ndownloader.figshare.com/files/13077299',
                      'decodedDNN-accuracy-VGG19-conv5_4.zip': 'https://ndownloader.figshare.com/files/13077302',
                      'decodedDNN-accuracy-VGG19-fc6.zip': 'https://ndownloader.figshare.com/files/13077305',
                      'decodedDNN-accuracy-VGG19-fc7.zip': 'https://ndownloader.figshare.com/files/13077308',
                      'decodedDNN-accuracy-VGG19-fc8.zip': 'https://ndownloader.figshare.com/files/13077311',
                      'decodedDNN-decoded-AlexNet-conv1.zip': 'https://ndownloader.figshare.com/files/13077257',
                      'decodedDNN-decoded-AlexNet-conv2.zip': 'https://ndownloader.figshare.com/files/13077260',
                      'decodedDNN-decoded-AlexNet-conv3.zip': 'https://ndownloader.figshare.com/files/13077263',
                      'decodedDNN-decoded-AlexNet-conv4.zip': 'https://ndownloader.figshare.com/files/13077266',
                      'decodedDNN-decoded-AlexNet-conv5.zip': 'https://ndownloader.figshare.com/files/13077173',
                      'decodedDNN-decoded-AlexNet-fc6.zip': 'https://ndownloader.figshare.com/files/13077176',
                      'decodedDNN-decoded-AlexNet-fc7.zip': 'https://ndownloader.figshare.com/files/13077179',
                      'decodedDNN-decoded-AlexNet-fc8.zip': 'https://ndownloader.figshare.com/files/13077182',
                      'decodedDNN-decoded-VGG19-conv1_1.zip': 'https://ndownloader.figshare.com/files/13077926',
                      'decodedDNN-decoded-VGG19-conv1_2.zip': 'https://ndownloader.figshare.com/files/13077962',
                      'decodedDNN-decoded-VGG19-conv2_1.zip': 'https://ndownloader.figshare.com/files/13077986',
                      'decodedDNN-decoded-VGG19-conv2_2.zip': 'https://ndownloader.figshare.com/files/13111850',
                      'decodedDNN-decoded-VGG19-conv3_1.zip': 'https://ndownloader.figshare.com/files/13078008',
                      'decodedDNN-decoded-VGG19-conv3_2.zip': 'https://ndownloader.figshare.com/files/13078023',
                      'decodedDNN-decoded-VGG19-conv3_3.zip': 'https://ndownloader.figshare.com/files/13078080',
                      'decodedDNN-decoded-VGG19-conv3_4.zip': 'https://ndownloader.figshare.com/files/13077653',
                      'decodedDNN-decoded-VGG19-conv4_1.zip': 'https://ndownloader.figshare.com/files/13077674',
                      'decodedDNN-decoded-VGG19-conv4_2.zip': 'https://ndownloader.figshare.com/files/13077704',
                      'decodedDNN-decoded-VGG19-conv4_3.zip': 'https://ndownloader.figshare.com/files/13077728',
                      'decodedDNN-decoded-VGG19-conv4_4.zip': 'https://ndownloader.figshare.com/files/13077731',
                      'decodedDNN-decoded-VGG19-conv5_1.zip': 'https://ndownloader.figshare.com/files/13077734',
                      'decodedDNN-decoded-VGG19-conv5_2.zip': 'https://ndownloader.figshare.com/files/13077737',
                      'decodedDNN-decoded-VGG19-conv5_3.zip': 'https://ndownloader.figshare.com/files/13077740',
                      'decodedDNN-decoded-VGG19-conv5_4.zip': 'https://ndownloader.figshare.com/files/13077752',
                      'decodedDNN-decoded-VGG19-fc6.zip': 'https://ndownloader.figshare.com/files/13077755',
                      'decodedDNN-decoded-VGG19-fc7.zip': 'https://ndownloader.figshare.com/files/13077758',
                      'decodedDNN-decoded-VGG19-fc8.zip': 'https://ndownloader.figshare.com/files/13077761',
                      'decodedDNN-rank-AlexNet-conv1.zip': 'https://ndownloader.figshare.com/files/13077185',
                      'decodedDNN-rank-AlexNet-conv2.zip': 'https://ndownloader.figshare.com/files/13077188',
                      'decodedDNN-rank-AlexNet-conv3.zip': 'https://ndownloader.figshare.com/files/13077191',
                      'decodedDNN-rank-AlexNet-conv4.zip': 'https://ndownloader.figshare.com/files/13077194',
                      'decodedDNN-rank-AlexNet-conv5.zip': 'https://ndownloader.figshare.com/files/13077197',
                      'decodedDNN-rank-AlexNet-fc6.zip': 'https://ndownloader.figshare.com/files/13077200',
                      'decodedDNN-rank-AlexNet-fc7.zip': 'https://ndownloader.figshare.com/files/13077203',
                      'decodedDNN-rank-AlexNet-fc8.zip': 'https://ndownloader.figshare.com/files/13077206',
                      'decodedDNN-rank-VGG19-conv1_1.zip': 'https://ndownloader.figshare.com/files/13083275',
                      'decodedDNN-rank-VGG19-conv1_2.zip': 'https://ndownloader.figshare.com/files/13083278',
                      'decodedDNN-rank-VGG19-conv2_1.zip': 'https://ndownloader.figshare.com/files/13083281',
                      'decodedDNN-rank-VGG19-conv2_2.zip': 'https://ndownloader.figshare.com/files/13083284',
                      'decodedDNN-rank-VGG19-conv3_1.zip': 'https://ndownloader.figshare.com/files/13083287',
                      'decodedDNN-rank-VGG19-conv3_2.zip': 'https://ndownloader.figshare.com/files/13083152',
                      'decodedDNN-rank-VGG19-conv3_3.zip': 'https://ndownloader.figshare.com/files/13083155',
                      'decodedDNN-rank-VGG19-conv3_4.zip': 'https://ndownloader.figshare.com/files/13083158',
                      'decodedDNN-rank-VGG19-conv4_1.zip': 'https://ndownloader.figshare.com/files/13083161',
                      'decodedDNN-rank-VGG19-conv4_2.zip': 'https://ndownloader.figshare.com/files/13083164',
                      'decodedDNN-rank-VGG19-conv4_3.zip': 'https://ndownloader.figshare.com/files/13083167',
                      'decodedDNN-rank-VGG19-conv4_4.zip': 'https://ndownloader.figshare.com/files/13083170',
                      'decodedDNN-rank-VGG19-conv5_1.zip': 'https://ndownloader.figshare.com/files/13083173',
                      'decodedDNN-rank-VGG19-conv5_2.zip': 'https://ndownloader.figshare.com/files/13083176',
                      'decodedDNN-rank-VGG19-conv5_3.zip': 'https://ndownloader.figshare.com/files/13083179',
                      'decodedDNN-rank-VGG19-conv5_4.zip': 'https://ndownloader.figshare.com/files/13083182',
                      'decodedDNN-rank-VGG19-fc6.zip': 'https://ndownloader.figshare.com/files/13083185',
                      'decodedDNN-rank-VGG19-fc7.zip': 'https://ndownloader.figshare.com/files/13083188',
                      'decodedDNN-rank-VGG19-fc8.zip': 'https://ndownloader.figshare.com/files/13083191',
                      'decodedDNN-true-AlexNet-conv1.zip': 'https://ndownloader.figshare.com/files/13110746',
                      'decodedDNN-true-AlexNet-conv2.zip': 'https://ndownloader.figshare.com/files/13111937',
                      'decodedDNN-true-AlexNet-conv3.zip': 'https://ndownloader.figshare.com/files/13110752',
                      'decodedDNN-true-AlexNet-conv4.zip': 'https://ndownloader.figshare.com/files/13110731',
                      'decodedDNN-true-AlexNet-conv5.zip': 'https://ndownloader.figshare.com/files/13110734',
                      'decodedDNN-true-AlexNet-fc6.zip': 'https://ndownloader.figshare.com/files/13110740',
                      'decodedDNN-true-AlexNet-fc7.zip': 'https://ndownloader.figshare.com/files/13110743',
                      'decodedDNN-true-AlexNet-fc8.zip': 'https://ndownloader.figshare.com/files/13110737',
                      'decodedDNN-true-VGG19-conv1_1.zip': 'https://ndownloader.figshare.com/files/13110797',
                      'decodedDNN-true-VGG19-conv1_2.zip': 'https://ndownloader.figshare.com/files/13110800',
                      'decodedDNN-true-VGG19-conv2_1.zip': 'https://ndownloader.figshare.com/files/13110803',
                      'decodedDNN-true-VGG19-conv2_2.zip': 'https://ndownloader.figshare.com/files/13110806',
                      'decodedDNN-true-VGG19-conv3_1.zip': 'https://ndownloader.figshare.com/files/13110812',
                      'decodedDNN-true-VGG19-conv3_2.zip': 'https://ndownloader.figshare.com/files/13110815',
                      'decodedDNN-true-VGG19-conv3_3.zip': 'https://ndownloader.figshare.com/files/13110755',
                      'decodedDNN-true-VGG19-conv3_4.zip': 'https://ndownloader.figshare.com/files/13110761',
                      'decodedDNN-true-VGG19-conv4_1.zip': 'https://ndownloader.figshare.com/files/13110764',
                      'decodedDNN-true-VGG19-conv4_2.zip': 'https://ndownloader.figshare.com/files/13110767',
                      'decodedDNN-true-VGG19-conv4_3.zip': 'https://ndownloader.figshare.com/files/13110770',
                      'decodedDNN-true-VGG19-conv4_4.zip': 'https://ndownloader.figshare.com/files/13110773',
                      'decodedDNN-true-VGG19-conv5_1.zip': 'https://ndownloader.figshare.com/files/13110776',
                      'decodedDNN-true-VGG19-conv5_2.zip': 'https://ndownloader.figshare.com/files/13110779',
                      'decodedDNN-true-VGG19-conv5_3.zip': 'https://ndownloader.figshare.com/files/13110782',
                      'decodedDNN-true-VGG19-conv5_4.zip': 'https://ndownloader.figshare.com/files/13110785',
                      'decodedDNN-true-VGG19-fc6.zip': 'https://ndownloader.figshare.com/files/13110788',
                      'decodedDNN-true-VGG19-fc7.zip': 'https://ndownloader.figshare.com/files/13110791',
                      'decodedDNN-true-VGG19-fc8.zip': 'https://ndownloader.figshare.com/files/13110794'}

    @property
    def _remote_files(self):
        return self.__remote_files

    def __init__(self, datastore=None, verbose=False, auto_download=False):
        super(DecodedDNN, self).__init__(datastore=datastore, verbose=verbose, auto_download=auto_download, default_dir='decodeddnn')

    def _get_files(self, mode=None, subject=None, net=None, layer=None, image=None):

        # Input processing
        if mode is None: raise RuntimeError('`mode` is required.')

        if subject is None: subject = DecodedDNN.__subjects
        subject = self.__listize(subject)

        if mode == 'true':
            subject = [None]

        if net is None: net = list(DecodedDNN.__nets.keys())
        net = self.__listize(net)

        net_dict = {}
        for nt in net:
            layer_tmp = DecodedDNN.__nets[nt] if layer is None else layer
            layer_tmp = self.__listize(layer_tmp)
            net_dict.update({nt : layer_tmp})

        if mode == 'decoded' or mode == 'true':
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
        elif mode == 'true':
            fpath = os.path.join(mode, net, layer, '%s-%s-%s-%s.mat' % (mode, net, layer, image))
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
            if 'feat' in d: return d['feat'].value.transpose(2, 1, 0)
            if 'accuracy' in d: return d['accuracy'].value.transpose(2, 1, 0)
            if 'rank' in d: return d['rank'].value.transpose(2, 1, 0)

        raise RuntimeError('Invalid data: %s' % fpath)

    def _download_file(self, fname):
        if fname in DecodedDNN.__remote_files:
            remote_file = fname
        else:
            fstr = fname.split('/')
            remote_file = 'decodedDNN-%s-%s-%s.zip' % (fstr[0], fstr[1], fstr[2])

        url = DecodedDNN.__remote_files[remote_file]

        private_link = '76f1e5e10fdfacc5ec12'
        url = '%s?private_link=%s' % (url, private_link)

        t = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        tempdir = os.path.join(self._datastore, '.temp-%s' % t)
        tempfile = os.path.join(self._datastore, remote_file)

        download_file(url, tempfile)

        # Unzip the file
        with zipfile.ZipFile(tempfile, 'r') as f:
            f.extractall(tempdir)

        # Move files
        for ds, ss, fs in os.walk(tempdir):
            if not fs: continue
            for f in fs:
                src = os.path.join(ds, f)
                trg = os.path.join(self._datastore,
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
