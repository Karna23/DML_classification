import os
import torch
from PIL import Image
from torchvision.transforms import functional as F
import copy
import json
import numpy

class Set(torch.utils.data.Dataset):
    def __init__(self, root, path, set_type, transform = None):
        
        self.root = root
        self.path = path
        self.transform = transform
        self.set_type = set_type if set_type != "gen" else "train"
        self.import_records()

    def import_records(self):
        folder_path = os.path.join(self.root, self.path, "{}".format(self.set_type))
        label_paths = [os.path.join(folder_path, elm) for elm in os.listdir(folder_path)]
        image_paths = []
        for i in range(len(label_paths)):
            image_paths += [os.path.join(label_paths[i], elm) for elm in os.listdir(label_paths[i])]
        self.records = []
        for elm in image_paths:
            if "\\" in elm:
                label = elm.split("\\")[-2]
            elif "/" in elm:
                label = elm.split("/")[-2]
            record = {"label": label, "img_path": elm}
            self.records.append(record)

        self.labels = numpy.unique([elm["label"] for elm in self.records])
        self.label_map = {}
        for cnt,elm in enumerate(self.labels):
            self.label_map[elm] = cnt
                    
    def nb_classes(self):
        return len(self.labels)

    def __len__(self):
        return len(self.records)
    
    def img_load(self, path):
        im = Image.open(path)
        if len(list(im.split())) == 1:
            im = im.convert('RGB')
        return im
    
    def __getitem__(self, index):
        record = self.records[index]
        img_path = record["img_path"]
        sample = {}
        sample["image"] = self.img_load(img_path)
        sample["label_str"] = record["label"]
        sample["label_int"] = self.label_map[record["label"]]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    
class InferenceSet(torch.utils.data.Dataset):
    def __init__(self,data_root,
                 inspection_file,
                 cfg,
                 transform = None):
        
        self.root = data_root
        self.inspection_file = inspection_file
        self.transform = transform
        self.import_records()
        
    def img_load(self, path):
            im = Image.open(path)
            if len(list(im.split())) == 1:
                im = im.convert('RGB')
            return im

    def import_records(self):
        self.records = []
        for key in self.inspection_file:
            for cnt, elm in enumerate(self.inspection_file[key]["detection"]):
                elm["source_path"] = self.inspection_file[key]["source_path"]
                elm["uniqueid"] = self.inspection_file[key]["uniqueid"]
                elm["lightposition"] = self.inspection_file[key]["lightposition"]
                elm["img_path"] = elm["crop_path"]
                del elm["crop_path"]
                self.records.append(elm)
    
    def __getitem__(self, index):
        sample = self.records[index]
        sample["image"] = self.img_load(sample["img_path"])
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return len(self.records)