import numpy as np
import os
import torch
import tqdm
import random
import config
import datasets

import models

from utils import l2_norm

assert torch.__version__.split('.')[0] == '1'

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

print('CUDA available: {}'.format(torch.cuda.is_available()))
cfg = config.load("./config/config_parkingbreaks.json")
start_epoch = 0
counter_device = "cpu"
cfg.resume = "./checkpoints/{}-{}-EmbSize{}-Loss{}-InpSize{}-{}-parallel{}/ckpt".format( 
                                                          "DML_parkingbreaks_1",
                                                          cfg.backbone,
                                                          cfg.embedding_size,
                                                          cfg.loss,
                                                          cfg.input_size,
                                                          cfg.optimizer,
                                                          cfg.data_parallel) # cfg.__dict__["dataset"]

# import dataset
os.chdir("datasets")
cfg.data_root = os.getcwd()
ds_gen, dl_gen = datasets.load(cfg, gen=True, val=False)

model = models.load(cfg, pretrained=False)

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
model.eval()

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

pbar = tqdm.tqdm(enumerate(dl_gen))

for batch_idx, data in pbar:
    with torch.no_grad():                      
        m = model(data["image"].squeeze().to(cfg.device))
        m = l2_norm(m)
        data["label_int"]
        if batch_idx == 0:
            embeddings = m
            labels_int = data["label_int"]
            labels_str = data["label_str"]
        else:
            embeddings = torch.cat((embeddings, m), dim = 0)
            labels_int = torch.cat((labels_int, data["label_int"]), dim = 0)
            labels_str = np.concatenate((labels_str, data["label_str"]), axis = 0)

    pbar.set_description(
            'Generate: [{}/{} ({:.0f}%)]'.format(
            batch_idx + 1, len(dl_gen),
            100. * (batch_idx+1) / len(dl_gen)))

# label mapping
label_map = {}
for elm in zip(labels_int, labels_str):
    label_map[elm[0].item()] = elm[1]

torch.save({'model_state_dict': model.state_dict(),
            'optimizer': checkpoint["optimizer"],
            'epoch': checkpoint["epoch"],
            'parallel_flag': cfg.data_parallel,
            "embeddings": embeddings,
            "labels_int": labels_int,
            "labels_str": labels_str,
            "label_map": label_map},
            '{}.pth'.format(cfg.resume))