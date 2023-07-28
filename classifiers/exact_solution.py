# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:00:24 2021

@author: tekin.evrim.ozmermer
"""
import torch
import sklearn.preprocessing

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, nb_classes)
    )
    T = torch.FloatTensor(T)  # .cuda()
    return T


class ExactSolution(torch.nn.Module):
    def __init__(self, cfg, embedding_collection,
                 labels_str, labels_int, label_map):
        super(ExactSolution, self).__init__()

        self.cfg = cfg
        self.raw_collection = embedding_collection
        self.labels_str = labels_str
        self.labels_bin = binarize(labels_int, labels_int.max())
        self.label_map = label_map
        self.linear = torch.nn.Linear(in_features=embedding_collection.shape[1],
                                      out_features=self.labels_bin.shape[1],
                                      bias=False)

    def calc_magnitude(self, input):
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)
        return norm

    def solve_exact(self):
        collection_inverse = torch.pinverse(l2_norm(self.raw_collection))
        self.W = torch.matmul(collection_inverse.to(self.cfg.device),
                              self.labels_bin.to(self.cfg.device))
        with torch.no_grad():
            self.linear.weight = torch.nn.Parameter(self.W.T)

    def forward(self, embedding):
        if len(embedding.shape) < 2:
            embedding = embedding.unsqueeze(0)
        out = self.linear(l2_norm(embedding))
        out = torch.where(out > 1, 2 - out, out)

        pred_single_int = int((1 - out).abs().argmin().to("cpu").numpy())
        pred_single_str = self.label_map[pred_single_int]
        out_dict = {
            'label': str(pred_single_str),
            'confidence': float(out.max().detach().to("cpu").numpy())
        }
        return out_dict