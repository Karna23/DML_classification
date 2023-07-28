import collections
import numpy as np
import os
import torch
import tqdm
import json

import random
import config
import datasets
import models
import classifiers
from utils import *

assert torch.__version__.split('.')[0] == '1'

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

print('CUDA available: {}'.format(torch.cuda.is_available()))
cfg = config.load("./config/config.json")
start_epoch = 0
counter_device = "cpu"
cfg.resume = "./checkpoints/{}-{}-EmbSize{}-Loss{}-InpSize{}-{}-parallel{}/ckpt".format(
    "DML_slot_nuts",
    cfg.backbone,
    cfg.embedding_size,
    cfg.loss,
    cfg.input_size,
    cfg.optimizer,
    cfg.data_parallel)  # cfg.__dict__["dataset"]

# import dataset
os.chdir("datasets")
cfg.data_root = os.getcwd()
ds_tr, dl_tr, ds_ev, dl_ev, ds_gen, dl_gen = datasets.load(cfg, val=True)
model = models.load(cfg, pretrained=False)



os.chdir("..")

# resume
checkpoint = None
if os.path.isfile("{}.pth".format(cfg.resume)):
    print('=> loading checkpoint:\n{}.pth'.format(cfg.resume))
    checkpoint = torch.load("{}.pth".format(cfg.resume), torch.device(cfg.device))
    if checkpoint['parallel_flag']:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
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

classifier = classifiers.load(cfg, checkpoint)
classifier.eval()

pbar = tqdm.tqdm(enumerate(dl_ev))
for batch_idx, data in pbar:
    with torch.no_grad():
        m = model(data["image"].to(cfg.device))
        m = l2_norm(m)

        if batch_idx == 0:
            embeddings = m
            labels_int = data["label_int"]
            labels_str = data["label_str"]
        else:
            embeddings = torch.cat((embeddings, m), dim=0)
            labels_int = torch.cat((labels_int, data["label_int"]), dim=0)
            labels_str = np.concatenate((labels_str, data["label_str"]), axis=0)

    pbar.set_description(
        'Generate: [{}/{} ({:.0f}%)]'.format(
            batch_idx + 1, len(dl_ev),
            100. * (batch_idx + 1) / len(dl_ev)))

pbar = tqdm.tqdm(enumerate(embeddings))

true_predictions = 0
false_predictions = 0
predictions = {}
for idx, emb in pbar:
    with torch.no_grad():
        out = classifier(emb)
        if out["label"] == labels_str[idx]:
            true_predictions += 1
        else:
            false_predictions += 1
        predictions[idx] = {"accurate": out["label"] == labels_str[idx],
                            "confidence": out["confidence"],
                            "label": out["label"],
                            "ground_truth": labels_str[idx]}
        pbar.set_description(
            'Evaluate: [{}/{} ({:.0f}%)]'.format(
            idx + 1, len(dl_ev),
            100. * (idx + 1) / len(dl_ev)))

print("true_predictions:", true_predictions)
print("false_predictions:", false_predictions)
print("precision: ", true_predictions/(true_predictions+false_predictions))

with open("predictions.json", "w") as fp:
    json.dump(predictions, fp, indent = 4)