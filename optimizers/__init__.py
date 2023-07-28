# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 13:54:04 2021

@author: tekin.evrim.ozmermer
"""
import torch
from sys import exit as EXIT
import math

def load(cfg, param_groups):
        
    if cfg.optimizer == "sgd":
        opt = torch.optim.SGD(param_groups, lr = float(cfg.lr))
        
    elif cfg.optimizer == 'adam': 
        opt = torch.optim.Adam(param_groups, lr = float(cfg.lr))
        
    elif cfg.optimizer == 'rmsprop':
        opt = torch.optim.RMSprop(param_groups, lr = float(cfg.lr))
        
    elif cfg.optimizer == 'adamw':
        opt = torch.optim.AdamW(param_groups, lr = float(cfg.lr))
        
    else:
        print('Given optimizer is wrong. Choose one of sgd, adam, rmsprop, adamw')
        EXIT(0)
        
    return opt