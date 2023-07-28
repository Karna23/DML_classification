# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 20:19:52 2021

@author: tekin.evrim.ozmermer
"""
import torch 
from .augmentations import TransformTrain, TransformEvaluate, TransformInference
from .base import Set, InferenceSet

def load(cfg, val = False, gen = False):
    if val:
        ds_tr = Set(cfg.data_root,
                    cfg.dataset,
                    set_type = "train",
                    transform = TransformTrain(cfg))
        
        dl_tr = torch.utils.data.DataLoader(ds_tr,
                                            batch_size = cfg.batch_size,
                                            collate_fn = collater,
                                            shuffle = True,
                                            num_workers = 0,
                                            drop_last = True,
                                            pin_memory = True)
        
        ds_ev = Set(cfg.data_root,
                    cfg.dataset,
                    set_type = "eval",
                    transform = TransformEvaluate(cfg))
        
        dl_ev = torch.utils.data.DataLoader(ds_ev,
                                            batch_size = 1,
                                            shuffle = False,
                                            num_workers = 0,
                                            drop_last = True,
                                            pin_memory = True)
        
        ds_gen = Set(cfg.data_root,
                    cfg.dataset,
                    set_type = "gen",
                    transform = TransformEvaluate(cfg))
        
        dl_gen = torch.utils.data.DataLoader(ds_gen,
                                             batch_size = int(cfg.batch_size*2),
                                             collate_fn = collater,
                                             shuffle = False,
                                             num_workers = 0,
                                             drop_last = False,
                                             pin_memory = True)
        
        return ds_tr, dl_tr, ds_ev, dl_ev, ds_gen, dl_gen
    
    elif gen:
        ds_gen = Set(cfg.data_root,
                    cfg.dataset,
                    set_type = "gen",
                    transform = TransformEvaluate(cfg))
        
        dl_gen = torch.utils.data.DataLoader(ds_gen,
                                             batch_size = int(cfg.batch_size*2),
                                             collate_fn = collater,
                                             shuffle = False,
                                             num_workers = 0,
                                             drop_last = False,
                                             pin_memory = True)
        return ds_gen, dl_gen
    
    else:
        ds_tr = Set(cfg.data_root,
                    cfg.dataset,
                    set_type = "train",
                    transform = TransformTrain(cfg))
        
        dl_tr = torch.utils.data.DataLoader(ds_tr,
                                            batch_size = cfg.batch_size,
                                            collate_fn = collater,
                                            shuffle = True,
                                            num_workers = 0,
                                            drop_last = True,
                                            pin_memory = True)
        return ds_tr, dl_tr
        
def load_folder(cfg, data_root, inspection_file):
    # for inference purpose
    # from folder of images to inference
    ds_inf = InferenceSet(data_root = data_root,
                          inspection_file = inspection_file,
                          cfg = cfg,
                          transform = TransformInference(cfg))
    
    dl_inf = torch.utils.data.DataLoader(ds_inf,
                                         batch_size = cfg.inference_batch_size,
                                         shuffle = False,
                                         num_workers = 0,
                                         drop_last = False,
                                         pin_memory = False)
    
    return dl_inf
    
def collater(data):
    # Function to pull single image
    # and put to batch. This is for Pytorch dataloader
    # to load input batches.
    keys = list(data[0].keys())
    
    out = {}
    for key in keys:
        try:
            data_piece = [s[key] for s in data]
            if key == "image":
                data_piece = torch.stack(data_piece, dim = 0)
            elif key in ["score","label_int","confidence"]:
                data_piece = torch.tensor(data_piece)
            else:
                pass
            out[key] = data_piece
        except:
            pass

    return out