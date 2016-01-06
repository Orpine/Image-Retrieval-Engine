from feat_extractor import *
from SdA import *
from scipy.spatial.distance import *
import json
class engine:
    
    def __init__(self):
        self.loaded = False
        caffe.set_mode_cpu()
        model_file = 'caffe_models/vgg_16/VGG_ILSVRC_16_layers_deploy.prototxt'
        pretrained_file = 'caffe_models/vgg_16/VGG_ILSVRC_16_layers.caffemodel'
        start = time.time()
        self.extractor = FeatExtractor(model_file, pretrained_file, oversample=False)
        print 'intitialization time:', time.time() - start
    
    def build(self):
        feat_extract(5000, self.extractor) # 5000 = batch_size

    def encode(self):
        X = np.load('feats_total_4096.npy')
        sda = SdAWrapper(X)
        np.save('sda_model', sda)
        y = sda.get_lowest_hidden_values(X)
        get_y = theano.function([], y)
        y_val = get_y()
        np.save('feats_total_64.npy', y_val)

    def init(self, load = False):
        if not load:
            build()
            encode()
        else:
            self.sda = np.load('sda_model.npy').all()
            self.feats_4096 = np.load('feats_total_4096.npy')
            self.feats_64 = np.load('feats_total_64.npy')
            self.filename = np.load('filename.npy')
            self.loaded = True
    
    def get_feat(self, img):
        return self.extractor.extract([img], ['fc7'])['fc7'][0]
    
    def get_feat_64(self, img):
        y = self.sda.get_lowest_hidden_values([img])
        get_y = theano.function([], y)
        y_val = get_y()     
        return y_val
    
    def predict(self, img, predict_on_64, metric):
        if not self.loaded:
            self.init(True)
        feat = self.get_feat(img)
        if predict_on_64:
            feat = self.get_feat_64(feat)
            dis = cdist(feat, self.feats_64, metric)
        else:
            dis = cdist([feat], self.feats_4096, metric)
        return json.dumps(dict(zip(range(100), self.filename[np.argsort(dis)[0][0:100]])), indent=4, separators=(',', ': '))