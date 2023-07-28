import tqdm
import random
import numpy as np
import torch, os
import copy

import json

from loss.proxy_anchor_vector import ProxyAnchorLoss as ProxyVector
from loss.proxy_anchor_features import ProxyAnchorFeatureLoss as ProxyFeature
from loss.others import Proxy_NCA, MultiSimilarityLoss, ContrastiveLoss, TripletLoss, NPairLoss

from net.resnet import Resnet18,Resnet34,Resnet50,Resnet101
from net.googlenet import googlenet
from net.bn_inception import bn_inception

import dataset
import utils_feature as ut_f
import utils_vector as ut_v

from config import cfg

print(cfg)

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
cfg.embedding_size = int(cfg.embedding_size)

device = cfg.device
counter_device = 'cpu'

# Directory for model directory
models_dir = cfg.models_dir+'/model_{}/dataset@{}_arch@{}_loss@{}_embedsize@{}_alpha@{}_margin@{}_optimizer@{}_batch@{}_decomposition@{}'.format(
                        cfg.dataset,cfg.dataset,cfg.model,cfg.loss,cfg.embedding_size, 
                        cfg.alpha, cfg.mrg, cfg.optimizer, cfg.batch_size, cfg.decomposition).replace('.','')

# Dataset Loader and Sampler
os.chdir('../recognition_datasets/')
data_root = os.getcwd()

# Build training set
trn_dataset = dataset.load(name = cfg.dataset,
                           root = data_root,
                           dpath = '/'+cfg.dataset,
                           mode = 'train',
                           transform = dataset.utils.make_transform(
                               is_train = True,
                               is_inception = (cfg.model == 'bn_inception'),
                               resize_overwrite = cfg.input_size))

dl_tr = torch.utils.data.DataLoader(trn_dataset,
                                    batch_size = cfg.batch_size,
                                    shuffle = True,
                                    num_workers = cfg.num_workers,
                                    drop_last = True,
                                    pin_memory = True)

# Build evaluation set <<
ev_dataset = dataset.load(name = cfg.dataset,
                          root = data_root,
                          dpath = '/'+cfg.dataset,
                          mode = 'test',
                          transform = dataset.utils.make_transform(
                              is_train = False, 
                              is_inception = (cfg.model == 'bn_inception'),
                              resize_overwrite = cfg.input_size))

dl_ev = torch.utils.data.DataLoader(ev_dataset,
                                    batch_size = cfg.batch_size,
                                    shuffle = False,
                                    num_workers = cfg.num_workers,
                                    pin_memory = True)

# ut.save_debug_images(cfg, models_dir, dl_tr, 'train')
# ut.save_debug_images(cfg, models_dir, dl_ev, 'test')

nb_classes = trn_dataset.nb_classes()

# print("nb_classes:", nb_classes)

# Import Backbone Model
model_embedding_size = int(cfg.embedding_size+int(cfg.embedding_size/cfg.feature_size)) if cfg.loss == "ProxyAnchorVectorSum" else int(cfg.embedding_size)

if cfg.model.find('googlenet')+1:
    model = googlenet(embedding_size=model_embedding_size, pretrained=True, is_norm=cfg.l2_norm, bn_freeze = cfg.bn_freeze)
elif cfg.model.find('bn_inception')+1:
    model = bn_inception(embedding_size=model_embedding_size, pretrained=True, is_norm=cfg.l2_norm, bn_freeze = cfg.bn_freeze)
elif cfg.model.find('resnet18')+1:
    model = Resnet18(embedding_size=model_embedding_size, pretrained=True, is_norm=cfg.l2_norm, bn_freeze = cfg.bn_freeze)
elif cfg.model.find('resnet34')+1:
    model = Resnet34(embedding_size=model_embedding_size, pretrained=True, is_norm=cfg.l2_norm, bn_freeze = cfg.bn_freeze)
elif cfg.model.find('resnet50')+1:
    model = Resnet50(embedding_size=model_embedding_size, pretrained=True, is_norm=cfg.l2_norm, bn_freeze = cfg.bn_freeze)
elif cfg.model.find('resnet101')+1:
    model = Resnet101(embedding_size=model_embedding_size, pretrained=True, is_norm=cfg.l2_norm, bn_freeze = cfg.bn_freeze)

# load model if exists from before
if os.path.isfile('../recognition_models'+cfg.resume+'/model_last.pth'):
    print('=> loading checkpoint:\n{}'.format('../recognition_models'+cfg.resume+'/model_last.pth'))
    checkpoint = torch.load('../recognition_models'+cfg.resume+'/model_last.pth',torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    print('=> No checkpoint found at:\n{}'.format('../recognition_models'+cfg.resume+'/model_last.pth'))
model = model.to(device)

if cfg.loss == 'ProxyAnchorFeature':
    if os.path.isfile('../proxies'+cfg.resume+'/feature_proxies.pth'):
        print('=> loading proxies:\n{}'.format('../proxies'+cfg.resume+'/feature_proxies.pth'))
        data = torch.load('../proxies'+cfg.resume+'/feature_proxies.pth')
        
        ProxyAnchorMult = data['proxies']
        initial_label_map = data['label_map']
        final_label_map = trn_dataset.label_map
        final_labels = sorted([elm for elm in final_label_map if elm not in initial_label_map] + [elm for elm in initial_label_map if elm not in final_label_map])
        final_label_map = {}
        for cnt,elm in enumerate(final_labels):
            final_label_map[elm] = cnt
        
        # print("final_label_map==initial_label_map:", final_label_map==initial_label_map)
        
        if not (final_label_map==initial_label_map):
            old_classes = list(initial_label_map.keys())
            new_classes = [elm for elm in list(final_label_map.keys()) if elm not in old_classes]
    
            nb_classes = len(old_classes) + len(new_classes)
            
            #create candidate proxies
            candprox_dataset = copy.deepcopy(ev_dataset)
            dl_cand = torch.utils.data.DataLoader(ev_dataset,
                                                  batch_size = 1,
                                                  shuffle = False,
                                                  num_workers = cfg.num_workers,
                                                  pin_memory = True)
            
            model.eval()
            candidate_proxies_dict = {}
            with torch.no_grad():
                print("Generating embedding...")
                for batch_id, batch in enumerate(dl_cand):
                    if batch[2][0] in new_classes:
                        input_ = batch[0].to(device)
                        out_ = model(input_).to(counter_device)
                        if batch[2][0] in candidate_proxies_dict:
                            candidate_proxies_dict[batch[2][0]] = torch.cat((candidate_proxies_dict[batch[2][0]], 
                                                                             out_), axis = 0)
                        else:
                            candidate_proxies_dict[batch[2][0]] = out_
            
            candidate_label_map = {}
            for cnt,key in enumerate(candidate_proxies_dict):
                candidate_label_map[key] = cnt
                if cnt == 0:
                    out = candidate_proxies_dict[key].mean(axis = 0).data.unsqueeze(0)
                else:
                    out = torch.cat((out,candidate_proxies_dict[key].mean(axis = 0).data.unsqueeze(0)), axis = 0)
            candidate_proxies = torch.tensor(out, requires_grad = True)
            
            criterion = ProxyFeature(nb_classes = nb_classes,
                                        sz_embed = cfg.embedding_size,
                                        sz_feat = cfg.feature_size,
                                        mrg = cfg.mrg,
                                        alpha = cfg.alpha,
                                        pop_task = 'add',
                                        ProxyAnchorMult = ProxyAnchorMult,
                                        initial_label_map = initial_label_map,
                                        candidate_proxies = candidate_proxies,
                                        candidate_label_map = candidate_label_map,
                                        final_label_map = final_label_map)
            proxies = criterion.ProxyAnchorMult
            ut_f.save_proxies(cfg, 'feature_proxies', proxies, final_label_map)
        else:
            criterion = ProxyFeature(nb_classes = nb_classes,
                                        sz_embed = cfg.embedding_size,
                                        sz_feat = cfg.feature_size,
                                        mrg = cfg.mrg,
                                        alpha = cfg.alpha,
                                        pop_task = 'create',
                                        ProxyAnchorMult = ProxyAnchorMult)
            proxies = criterion.ProxyAnchorMult
            ut_f.save_proxies(cfg, 'feature_proxies', proxies, final_label_map)
    else:
        print('=> No proxies found at:\n{}'.format('../proxies'+cfg.resume+'/proxies.pth'))
        criterion = ProxyFeature(nb_classes = nb_classes,
                                    sz_embed = cfg.embedding_size,
                                    sz_feat = cfg.feature_size,
                                    mrg = cfg.mrg,
                                    alpha = cfg.alpha,
                                    pop_task = 'create')
        final_label_map = trn_dataset.label_map
        proxies = criterion.ProxyAnchorMult
        ut_f.save_proxies(cfg, 'feature_proxies', proxies, final_label_map)
        
elif cfg.loss == 'ProxyAnchorVector' or cfg.loss == 'ProxyAnchorVectorSum':
    sz_embed = cfg.embedding_size if cfg.loss == 'ProxyAnchorVector' else cfg.feature_size
    if os.path.isfile('../proxies'+cfg.resume+'/vector_proxies.pth'):
        print('=> loading proxies:\n{}'.format('../proxies'+cfg.resume+'/vector_proxies.pth'))
        data = torch.load('../proxies'+cfg.resume+'/vector_proxies.pth')
        
        initial_proxies = data['proxies']
        initial_label_map = data['label_map']
        final_label_map = trn_dataset.label_map
        # print("Before // Final Labels:", [elm for elm in final_label_map if elm not in initial_label_map])
        # print("Before // Initial Labels:", [elm for elm in initial_label_map if elm not in final_label_map])
        final_labels = sorted([elm for elm in final_label_map if elm not in initial_label_map] + [elm for elm in initial_label_map if elm not in final_label_map] + [elm for elm in initial_label_map if elm in final_label_map])
        print("After // Final Labels Length:", len(final_labels))
        final_label_map = {}
        for cnt,elm in enumerate(final_labels):
            final_label_map[elm] = cnt
        
        # print("final_label_map==initial_label_map:", sorted(list(final_label_map.keys())))
         
        if not (sorted(list(final_label_map.keys()))==sorted(list(initial_label_map.keys()))):
            old_classes = list(initial_label_map.keys())
            new_classes = [elm for elm in list(final_label_map.keys()) if elm not in old_classes]
    
            nb_classes = len(old_classes) + len(new_classes)
    
            #create candidate proxies
            candprox_dataset = copy.deepcopy(ev_dataset)
            dl_cand = torch.utils.data.DataLoader(ev_dataset,
                                                  batch_size = 1,
                                                  shuffle = False,
                                                  num_workers = cfg.num_workers,
                                                  pin_memory = True)
            
            model.eval()
            candidate_proxies_dict = {}
            with torch.no_grad():
                print("Generating embedding...")
                for batch_id, batch in enumerate(dl_cand):
                    if batch[2][0] in new_classes:
                        input_ = batch[0].to(device)
                        out_ = model(input_).to(counter_device)
                        if batch[2][0] in candidate_proxies_dict:
                            candidate_proxies_dict[batch[2][0]] = torch.cat((candidate_proxies_dict[batch[2][0]], 
                                                                             out_), axis = 0)
                        else:
                            candidate_proxies_dict[batch[2][0]] = out_
            
            candidate_label_map = {}
            for cnt,key in enumerate(candidate_proxies_dict):
                candidate_label_map[key] = cnt
                if cnt == 0:
                    out = candidate_proxies_dict[key].mean(axis = 0).data.unsqueeze(0)
                else:
                    out = torch.cat((out,candidate_proxies_dict[key].mean(axis = 0).data.unsqueeze(0)), axis = 0)
            candidate_proxies = torch.tensor(out, requires_grad = True)
            
            criterion = ProxyVector(nb_classes = nb_classes,
                                    sz_embed = cfg.feature_size if cfg.loss == "ProxyAnchorVectorSum" else cfg.embedding_size,
                                    mrg = cfg.mrg,
                                    alpha = cfg.alpha,
                                    pop_task = 'add',
                                    initial_proxies = initial_proxies,
                                    initial_label_map = initial_label_map,
                                    candidate_proxies = candidate_proxies,
                                    candidate_label_map = candidate_label_map,
                                    final_label_map = final_label_map)
            proxies = criterion.proxies
            ut_v.save_proxies(cfg, 'vector_proxies', proxies, final_label_map)
        else:
            criterion = ProxyVector(nb_classes = nb_classes,
                                    sz_embed = cfg.feature_size if cfg.loss == "ProxyAnchorVectorSum" else cfg.embedding_size,
                                    mrg = cfg.mrg,
                                    alpha = cfg.alpha,
                                    pop_task = 'create',
                                    initial_proxies = initial_proxies)
    else:
        final_label_map = trn_dataset.label_map
        nb_classes = len(final_label_map)
        criterion = ProxyVector(nb_classes = nb_classes,
                                sz_embed = cfg.feature_size if cfg.loss == "ProxyAnchorVectorSum" else cfg.embedding_size,
                                mrg = cfg.mrg,
                                alpha = cfg.alpha,
                                pop_task = 'create',
                                decompose = cfg.decomposition,
                                initial_proxies = None)
        proxies = criterion.proxies
        ut_v.save_proxies(cfg, 'vector_proxies', proxies, final_label_map)
        
elif cfg.loss == 'ProxyNCA':
    criterion = Proxy_NCA(nb_classes = nb_classes, sz_embed = cfg.embedding_size).to(counter_device)
elif cfg.loss == 'MS':
    criterion = MultiSimilarityLoss().to(counter_device)
elif cfg.loss == 'Contrastive':
    criterion = ContrastiveLoss().to(counter_device)
elif cfg.loss == 'Triplet':
    criterion = TripletLoss().to(counter_device)
elif cfg.loss == 'NPair':
    criterion = NPairLoss().to(counter_device)

# Train Parameters
param_groups = [
                {'params': list(set(model.parameters()).difference(set(model.model.embedding.parameters())))},
                {'params': model.model.embedding.parameters()}
               ]

# if cfg.loss in ['ProxyAnchorVectorSum', 'ProxyAnchorVector', 'ProxyAnchorFeature']:
#     param_groups.append({'params': criterion.proxies, 'lr':float(cfg.lr) * 100})

# Optimizer Setting
if cfg.optimizer == 'sgd':
    opt = torch.optim.SGD(param_groups, lr=float(cfg.lr), weight_decay = cfg.weight_decay, momentum = 0.9, nesterov=True)
elif cfg.optimizer == 'adam': 
    opt = torch.optim.Adam(param_groups, lr=float(cfg.lr), weight_decay = cfg.weight_decay)
elif cfg.optimizer == 'rmsprop':
    opt = torch.optim.RMSprop(param_groups, lr=float(cfg.lr), alpha=0.9, weight_decay = cfg.weight_decay, momentum = 0.9)
elif cfg.optimizer == 'adamw':
    opt = torch.optim.AdamW(param_groups, lr=float(cfg.lr), weight_decay = cfg.weight_decay)
    
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.lr_decay_step, gamma = cfg.lr_decay_gamma)

print("Training parameters: {}".format(vars(cfg)))
print("Training for {} epochs.".format(cfg.epochs))
losses_list = []
best_recall=[0]
best_epoch = 0

model.train()
# if device == "cuda":
#     model = torch.nn.DataParallel(model)

def reshape_sum_weight_emb(x,f_size):
    out = torch.zeros(x.shape[0],f_size).to(cfg.device)
    if cfg.weighted_sum:
        weights = ut_v.l2_norm(x[:,-4:])
    else:
        weights = torch.ones(x[:,-4:].shape).detach().to(cfg.device)
    for cnt,col in enumerate(range(0,x.shape[1]-int(cfg.embedding_size/f_size),f_size)):
        out = out + x[:,col:col+f_size]*weights[:,cnt].unsqueeze(1)
    return out

for epoch in range(0, cfg.epochs):
    model.train()
    bn_freeze = cfg.bn_freeze
    if bn_freeze:
        modules = model.model.modules()
        for m in modules: 
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

    losses_per_epoch = []

    pbar = tqdm.tqdm(enumerate(dl_tr))

    for batch_idx, (x, y, y_str) in pbar:                         
        m = model(x.squeeze().to(device))
        
        if cfg.loss == "ProxyAnchorVectorSum":
            m = reshape_sum_weight_emb(m,cfg.feature_size)
        
        if cfg.loss == "ProxyAnchorFeature":
            loss = criterion.run(m.to(counter_device), y.squeeze().to(counter_device))
        else:
            loss = criterion(m.to(counter_device), y.squeeze().to(counter_device))
        opt.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_value_(model.parameters(), 10)
        if cfg.loss == 'Proxy_Anchor':
            torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)

        losses_per_epoch.append(loss.data.cpu().numpy())
        opt.step()

        pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx + 1, len(dl_tr),
                100. * (batch_idx+1) / len(dl_tr),
                loss.item()))
        
    losses_list.append(np.mean(losses_per_epoch))
    scheduler.step()
    
    if epoch >= 0 and epoch%cfg.epoch_interval==0:
        with torch.no_grad():
            # print("**Evaluating...**")
            if cfg.loss == 'ProxyAnchorFeature':
                Recalls = ut_f.evaluate_cos(model, dl_ev, 8, device, cfg.feature_size, cfg.embedding_size)
            else:
                Recalls = ut_v.evaluate_cos(model, dl_ev, 8, device)
        #last model save
        try:
            os.mkdir('/'.join(models_dir.split('/')[0:3]))
        except:
            pass
        try:
            os.mkdir(models_dir)
        except:
            pass
        
        torch.save({'model_state_dict':model.state_dict()}, '{}/model_last.pth'.format(models_dir))
        with open('{}/logs_last_results.txt'.format(models_dir), 'w') as f:
            f.write('Last Epoch: {}\n'.format(epoch))
            for i in range(4):
                f.write("Last Recall@{}: {:.4f}\n".format(10**i, Recalls[i] * 100))

        # Best model save
        if best_recall[0] < Recalls[0]:
            best_recall = Recalls
            best_epoch = epoch
            if not os.path.exists('{}'.format(models_dir)):
                os.makedirs('{}'.format(models_dir))
            torch.save({'model_state_dict':model.state_dict()}, '{}/model_best.pth'.format(models_dir))
            with open('{}/logs_best_results.txt'.format(models_dir), 'w') as f:
                f.write('Best Epoch: {}\n'.format(best_epoch))
                for i in range(4):
                    f.write("Best Recall@{}: {:.4f}\n".format(10**i, best_recall[i] * 100))
                    
os.chdir('../recognition_production/')
#status check
success = 0
while not success:
    try:
        with open('./status.json','r') as f:
            status_dict = json.load(f)
            status_dict["evaluate"] = 0
            status_dict["train"] = 1
        success = 1
    except Exception as e:
        print(e)
        pass
        
success = 0
while not success:
    try:
        with open('./status.json','w') as f:
            json.dump(status_dict, f)
        success = 1
    except Exception as e:
        print(e)
        pass