"""
FeatExtractor is a feature extraction specialization of Net.
"""

import os.path
import sys


caffe_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'caffe', 'python')
if caffe_path not in sys.path:
    sys.path.insert(0, caffe_path)


import numpy as np
import caffe
from caffe_io import transform_image
import time


class FeatExtractor(caffe.Net):

    """
    Calls caffe_io to convert video/images
    and extract embedding features
    """

    def __init__(self, model_file, pretrained_file, img_dim=256,
                 crop_dim=224, mean=[103.939, 116.779, 123.68], oversample=False):
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)
        self.img_dim = img_dim
        self.crop_dim = crop_dim
        self.mean = mean
        self.oversample = oversample
        self.batch_size = 64  # hard coded, same as oversample patches

    def extract(self, imgs, blobs=['fc6', 'fc7']):
        feats = {}
        for blob in blobs:
            feats[blob] = []
        for img in imgs:
            data = transform_image(
                img, self.oversample, self.mean, self.img_dim, self.crop_dim)
            # Use forward all to do the padding
            out = self.forward_all(**{self.inputs[0]: data, 'blobs': blobs})
            for blob in blobs:
                feat = out[blob]
                if self.oversample:
                    feat = feat.reshape(
                        (len(feat) / self.batch_size, self.batch_size, -1))
                    feat = feat.mean(1)
                feats[blob].append(feat.flatten())
        return feats

    def _process_batch(self, data, feats, blobs):
        if data is None:
            return
        out = self.forward_all(**{self.inputs[0]: data, 'blobs': blobs})
        for blob in blobs:
            feat = out[blob]
            feat = feat.reshape((len(feat) / data.shape[0], data.shape[0], -1))
            for i in xrange(data.shape[0]):
                feats[blob].append(feat[:, i, :].flatten())

    def extract_batch(self, imgs, blobs=['fc6', 'fc7']):
        if self.oversample:   # Each oversampled image is a batch
            return self.extract(imgs, blobs)
        feats = {}
        for blob in blobs:
            feats[blob] = []
        data = None
        for img in imgs:
            if data is None:
                data = transform_image(
                    img, self.oversample, self.mean, self.img_dim, self.crop_dim)
            else:
                data = np.vstack((data, transform_image(
                    img, self.oversample, self.mean, self.img_dim, self.crop_dim)))
            if data.shape[0] == self.batch_size:
                self._process_batch(data, feats, blobs)
                data = None
        self._process_batch(data, feats, blobs)
        return feats


def get_image_names():
    import imghdr
    image_names = []
    for dirname in os.listdir('Images'):
        dirname = 'Images/' + dirname
        if os.path.isdir(dirname):
            for img in os.listdir(dirname):
                filename = dirname + '/' + img
                if imghdr.what(filename) is not None:
                    image_names.append(filename)
    return image_names


def feat_extract(batch, extractor):
    from caffe_io import load_image

    img_names = get_image_names()
    np.save('filename.npy', np.array(img_names))
    print len(img_names)
    imgs = []
    feats_ret = np.ndarray((0, 4096), dtype='f')
    for i in xrange(len(img_names)):
        if i % batch == 0 and i != 0:
            start = time.time()
            feats = extractor.extract_batch(imgs, ['fc7'])['fc7']
            print 'batch extraction:', time.time() - start
            np.save('feats_temp_' + str(i), feats)
            feats_ret = np.append(feats_ret, feats, axis=0)
            imgs = []
        img_name = img_names[i]
        img = load_image(img_name)
        imgs.append(img)
    if imgs != []:
        start = time.time()
        feats = extractor.extract_batch(imgs, ['fc7'])['fc7']
        print 'batch extraction:', time.time() - start
        np.save('feats_temp_' + str(len(img_names)), feats)
        feats_ret = np.append(feats_ret, feats, axis=0)
    np.save('feats_total_4096.npy', feats_ret)
    return feats_ret



if __name__ == '__main__':
    caffe.set_mode_cpu()
    model_file = 'caffe_models/vgg_16/VGG_ILSVRC_16_layers_deploy.prototxt'
    pretrained_file = 'caffe_models/vgg_16/VGG_ILSVRC_16_layers.caffemodel'
    start = time.time()
    extractor = FeatExtractor(model_file, pretrained_file, oversample=False)
    print 'intitialization time:', time.time() - start

    feat_extract(100)
