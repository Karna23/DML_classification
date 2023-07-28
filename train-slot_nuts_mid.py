import collections
import numpy as np
import os
import torch
import tqdm
import random
import config
import datasets
import optimizers

import loss
import models

from utils import *

assert torch.__version__.split('.')[0] == '1'

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

print('CUDA available: {}'.format(torch.cuda.is_available()))
cfg = config.load("./config/config-slotnuts_mid_dml.json")
start_epoch = 0
counter_device = "cpu"
cfg.resume = "./checkpoints/{}-{}-EmbSize{}-Loss{}-InpSize{}-{}-parallel{}/ckpt".format( 
                                                          "DML_slot_nuts_mid",
                                                          cfg.backbone,
                                                          cfg.embedding_size,
                                                          cfg.loss,
                                                          cfg.input_size,
                                                          cfg.optimizer,
                                                          cfg.data_parallel) # cfg.__dict__["dataset"]

# import dataset
os.chdir("datasets")
cfg.data_root = os.getcwd()
ds_tr, dl_tr, ds_ev, dl_ev, ds_gen, dl_gen = datasets.load(cfg, val=True)
model = models.load(cfg, pretrained = True)

os.chdir("..")

# resume
checkpoint = None
if os.path.isfile("{}.pth".format(cfg.resume)):
    print('=> loading checkpoint:\n{}.pth'.format(cfg.resume))
    checkpoint = torch.load("{}.pth".format(cfg.resume),torch.device(cfg.device))
    if checkpoint['parallel_flag']:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint['model_state_dict'], strict = True)
    else:
        model.load_state_dict(checkpoint['model_state_dict'], strict = True)
    start_epoch = checkpoint['epoch']
else:
    try:
        os.mkdir("/".join(cfg.resume.split("/")[0:2]))
    except:
        pass
    
    try:
        os.mkdir("/".join(cfg.resume.split("/")[0:3]))
    except:
        pass

if cfg.data_parallel:
    if checkpoint != None:
        if not checkpoint['parallel_flag']:
            model = torch.nn.DataParallel(model)
    else:
        model = torch.nn.DataParallel(model)
model.to(cfg.device)
criterion = loss.load(cfg, model, ds_tr, ds_gen)

param_groups = [
                {'params': model.parameters()}
               ]

# if cfg.loss == 'ProxyAnchor':
#     param_groups.append({'params': criterion.proxies, 'lr':float(cfg.lr) * 100})

opt = optimizers.load(cfg, param_groups)
# if checkpoint:
#     opt.load_state_dict(checkpoint['optimizer'])
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.lr_decay_step, gamma = cfg.lr_decay_gamma)

best_recall=[0]
best_epoch = 0
for epoch in range(start_epoch, cfg.epochs):
    model.train()
    bn_freeze = cfg.bn_freeze
    if bn_freeze:
        if cfg.data_parallel:
            modules = model.module.model.modules()
        else:
            modules = model.model.modules()
        for m in modules: 
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

    losses_per_epoch = []

    pbar = tqdm.tqdm(enumerate(dl_tr))

    for batch_idx, data in pbar:                         
        m = model(data["image"].squeeze().to(cfg.device))
        
        loss = criterion(m.to(counter_device), data["label_int"].squeeze().to(counter_device))
        opt.zero_grad()
        loss.backward()
        
        if cfg.loss == 'ProxyAnchor':
            torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)

        opt.step()

        pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx + 1, len(dl_tr),
                100. * (batch_idx+1) / len(dl_tr),
                loss.item()))
        
    scheduler.step()
    
    if epoch >= 0 and epoch%cfg.epoch_interval==0:
        with torch.no_grad():
            print("**Evaluating...**")
            Recalls = evaluate_cos(model, dl_ev, 4, cfg.device)

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer': opt.state_dict(),
                    'epoch': epoch,
                    'parallel_flag': cfg.data_parallel},
                    '{}.pth'.format(cfg.resume))
        
        with open('{}.txt'.format(cfg.resume.replace("ckpt","last_results")), 'w') as f:
            f.write('Last Epoch: {}\n'.format(epoch))
            for i in range(4):
                f.write("Last Recall@{}: {:.4f}\n".format(10**i, Recalls[i] * 100))

        # Best model save
        if best_recall[0] < Recalls[0]:
            best_recall = Recalls
            best_epoch = epoch

            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer': opt.state_dict(),
                        'epoch': epoch,
                        'parallel_flag': cfg.data_parallel},
                        '{}_best.pth'.format(cfg.resume))
            
            with open('{}.txt'.format(cfg.resume.replace("ckpt","best_results")), 'w') as f:
                f.write('Best Epoch: {}\n'.format(best_epoch))
                for i in range(4):
                    f.write("Best Recall@{}: {:.4f}\n".format(10**i, best_recall[i] * 100))