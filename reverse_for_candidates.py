import torch
import numpy as np
import sys, os

from config import cfg
import utils_vector as ut_v
import utils_feature as ut_f

device = cfg.device
counter_device = 'cpu'

class classifier_layer(torch.nn.Module):
    def __init__(self, embedding_collection,
                       labels_str, labels_int, labels_bin, label_map,
                       numof_new_classes = 0):
        super(classifier_layer, self).__init__()
        
        self.raw_collection = embedding_collection
        self.labels_str = labels_str
        self.labels_int = labels_int
        self.labels_bin = labels_bin
        self.label_map = label_map
        self.linear = torch.nn.Linear(in_features = embedding_collection.shape[1], out_features = 20, bias = False)
        self.retrain = False
    
    def l2norm(self,raw_collection):
        if cfg.loss == 'ProxyAnchorFeature':
            return ut_f.l2_norm(raw_collection)
        else:
            return ut_v.l2_norm(raw_collection)
    
    def calc_magnitude(self,input):
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)
        return norm
    
    def solve_exact(self):
        FC = self.raw_collection
        FC = self.l2norm(self.raw_collection)
        FCpi = torch.pinverse(FC)
        print('FC shape', FC.shape, 'FCpi shape', FCpi.shape)
        self.W = torch.matmul(FCpi,self.labels_bin)
        print("W shape", self.W.T.shape)
        print('linear weight shape', self.linear.weight.shape)
        with torch.no_grad():
            self.linear.weight = torch.nn.Parameter(self.W.T)

    def create_candidates(self, new_classes = [0,0]):
        new_classes = np.array(new_classes)
        
        self.labels_bin_forreverse = torch.cat((self.labels_bin,0.014*torch.rand(self.labels_bin.shape[0],new_classes.shape[1])),dim=1)
        self.solve_exact_forreverse()
        new_classes = torch.cat((torch.zeros(new_classes.shape[0],labels_bin.shape[1]),torch.tensor(new_classes)), dim = 1)
        self.W_inv = torch.pinverse(self.W_forreverse)
        self.candidates = torch.matmul(new_classes,self.W_inv).detach()
        with torch.no_grad():
            self.candidate_test_output = np.around(self.linear_forreverse(self.l2norm(self.candidates)).numpy(),2)

    def solve_exact_forreverse(self):
        FC = self.raw_collection
        FC = self.l2norm(self.raw_collection)
        FCpi = torch.pinverse(FC)
        print('FC shape', FC.shape, 'FCpi shape', FCpi.shape)
        self.W_forreverse = torch.matmul(FCpi,self.labels_bin_forreverse)
        print("W shape", self.W_forreverse.T.shape)
        with torch.no_grad():
            self.linear_forreverse = torch.nn.Linear(in_features  = self.linear.in_features,
                                                     out_features = self.linear.out_features, bias = False)
            self.linear_forreverse.weight = torch.nn.Parameter(self.W_forreverse.T)

    def forward(self,embedding):
        embedding_norm = embedding.unsqueeze(0)
        magnitude = self.calc_magnitude(embedding.unsqueeze(0)).numpy()
        embedding_norm = self.l2norm(embedding.unsqueeze(0))
        out = self.linear(embedding_norm)
        out = torch.where(out>1, 2-out, out)
        if self.retrain:
            out_addition = self.linear_addition(embedding_norm)
            out = out + out_addition
        pred_single_int = int(out.argmax().to(counter_device).numpy())
        pred_single_str = self.label_map[pred_single_int]
        out_dict = {'prediction_int': pred_single_int,
                    'prediction_str': str(pred_single_str),
                    'similarity_confidence': float(out.max().to(counter_device).detach().numpy()),
                    'entropy': torch.distributions.Categorical(out).entropy().to(counter_device).detach().numpy(),
                    'magnitude': magnitude,
                    'out': out.to(counter_device).detach().numpy()}
        print(str(pred_single_str))
        return out_dict
    
    def test_in_itself(self):
        raw_collection_norm = self.l2norm(self.raw_collection)
        out_asis = self.linear(raw_collection_norm)
        print('Test in itself -> Before:',(self.labels_bin-out_asis).max(),(self.labels_bin-out_asis).min())
        if self.retrain:
            out_addition = self.linear_addition(raw_collection_norm)
            out_asis = out_asis + out_addition
        print('Test in itself -> After:',(self.labels_bin-out_asis).max(),(self.labels_bin-out_asis).min())
        return out_asis
    
os.chdir('../embedding_collections/')
data_root = os.getcwd()

# Import data
if os.path.isfile('.'+cfg.resume+'/embedding_collection.pth'):
    print('=> loading checkpoint:\n{}'.format('.'+cfg.resume+'/embedding_collection.pth'))
    embedding_data = torch.load('.'+cfg.resume+'/embedding_collection.pth',map_location=torch.device(device))
    embedding = embedding_data['embedding']
    label_map = embedding_data['label_map']
    labels_int = embedding_data['labels_int']
    labels_str = embedding_data['labels_str']
else:
    print('=> No embedding collection found at:\n{}'.format(cfg.resume))
    sys.exit(0)

# binarize classes
labels_bin = np.zeros((labels_int.shape[0],len(label_map)))
for cnt,elm in enumerate(labels_int):
    labels_bin[cnt,elm] = 1

labels_bin = torch.from_numpy(labels_bin).float().to(counter_device)

# Classifier
model_cls = classifier_layer(embedding_collection = embedding.to(counter_device),
                             labels_str = labels_str,
                             labels_int = labels_int,
                             labels_bin = labels_bin,
                             label_map  = label_map)
                             
model_cls = model_cls.to(counter_device)

model_cls.solve_exact()

model_cls.create_candidates(new_classes = [[0,1,0,0,0]])
candidates1 = model_cls.candidates
print("Candidate test output: \n", model_cls.candidate_test_output)

model_cls.create_candidates(new_classes = [[0,1],[1,0],[1,1],[0,0]])
candidates2 = model_cls.candidates
print("Candidate test output: \n", model_cls.candidate_test_output)

gg = torch.nn.functional.linear(model_cls.l2norm(candidates1),model_cls.l2norm(embedding)).numpy()
print(gg)
# model_cls.raw_collection = None #Flush collection from memory because it is already used in training.