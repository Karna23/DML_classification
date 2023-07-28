# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:00:24 2021

@author: tekin.evrim.ozmermer
"""
import torch
import numpy as np


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output


class KNN(torch.nn.Module):
    def __init__(self, number_of_neighbours,
                 embedding_collection,
                 labels_str, labels_int, label_map):
        super(KNN, self).__init__()
        self.raw_collection = embedding_collection
        self.labels_str = labels_str
        self.labels_int = labels_int.cpu().numpy()
        self.label_map = label_map
        self.K = number_of_neighbours

    def forward(self, embedding):
        cos_sim = torch.nn.functional.linear(l2_norm(self.raw_collection),
                                             l2_norm(embedding.unsqueeze(0)).squeeze(0))
        cos_sim_topK = cos_sim.topk(1 + self.K)
        indexes = cos_sim_topK[1][1:self.K].cpu().numpy().tolist()
        probs = cos_sim_topK[0][1:self.K].cpu().numpy()

        preds_int = np.array([self.labels_int[i] for i in indexes])

        unqs, counts = np.unique(preds_int, return_counts=True)
        pred_single_int = unqs[np.argmax(counts)]
        pred_single_str = self.label_map[pred_single_int.item()]

        # neighbour_confidence = np.max(counts)/np.sum(counts)
        confidence = probs[np.where(preds_int == pred_single_int)][0]

        out_dict = {
            'label': str(pred_single_str),
            'confidence': float(confidence)
        }
        return out_dict