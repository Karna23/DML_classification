import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools
import numpy as np
from sys import exit as EXIT

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T)#.cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class ProxyAnchorLoss(torch.nn.Module):
    def __init__(self, nb_classes,
                 sz_embed,
                 mrg = 0.1,
                 alpha = 32,
                 pop_task = 'create',
                 initial_proxies = None,
                 initial_label_map = None,
                 candidate_label_map = None,
                 candidate_proxies = None,
                 final_label_map = None,
                 decompose = True):
        
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        dss = 1-(sz_embed/nb_classes)
        if dss<=0:
            dss = 0.01
        if pop_task == 'create':
            if initial_proxies==None:
                print('Task: ADD, Error: Input needed -> initial_proxies')
                base_proxies = initial_proxies
                torch.nn.init.kaiming_normal_(base_proxies, mode='fan_out')
                
            else:
                base_proxies = initial_proxies
                if decompose:
                    print("--> Decomposition is starting...")
                    pop_create = ProxyOperations(base_proxies = base_proxies,
                                                 task = pop_task)
                    proxies = pop_run(pop_create, max_iter = 100,
                                      desired_sim_score = dss, lr = 0.1)
                    print("--> Decomposition is done.")
                else:
                    print("--> Decomposition is deactivated.")
                    proxies = base_proxies
                    
                    try:
                        del pop_create
                        torch.cuda.empty_cache()
                    except:
                        pass
                
        elif pop_task == 'add':
            if initial_proxies==None:
                # print('Task: ADD, Error: Input needed -> initial_proxies')
                EXIT(0)
            if initial_label_map==None:
                # print('Task: ADD, Error: Input needed -> initial_label_map')
                EXIT(0)
            if candidate_proxies==None:
                # print('Task: ADD, Error: Input needed -> candidate_proxies')
                EXIT(0)
            if candidate_label_map==None:
                # print('Task: ADD, Error: Input needed -> candidate_label_map')
                EXIT(0)
            if final_label_map==None:
                # print('Task: ADD, Error: Input needed -> final_label_map')
                EXIT(0)
                
            else:
                pop_add = ProxyOperations(base_proxies = initial_proxies,
                                          initial_label_map = initial_label_map,
                                          candidate_label_map = candidate_label_map,
                                          candidate_proxies = candidate_proxies,
                                          final_label_map = final_label_map,
                                          task = pop_task)
                proxies = pop_run(pop_add, max_iter = 100,
                                  desired_sim_score = dss, lr = 0.1)
                
                pop_enhance = ProxyOperations(base_proxies = proxies,
                                              task = 'enhance')
                proxies = pop_run(pop_enhance, max_iter = 20,
                                  desired_sim_score = dss, lr = 0.1)
                
                del pop_add
                del pop_enhance
                torch.cuda.empty_cache()
         
        self.proxies = proxies
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
    def forward(self, X, T):
        P = self.proxies
        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        return loss

class ProxyOperations(torch.nn.Module):
    def __init__(self, base_proxies = None,
                 initial_label_map = None,
                 candidate_label_map = None,
                 candidate_proxies = None,
                 final_label_map = None,
                 task = 'create'):
        super(ProxyOperations, self).__init__()
        #tasks = create, add, enhance
        self.task = task
        if task in ['enhance', 'create']:
            self.base_proxies = torch.nn.Parameter(base_proxies)
        else:
            self.base_proxies = torch.tensor(base_proxies, requires_grad = False)
        if task in ['add']:
            self.candidate_proxies = torch.nn.Parameter(candidate_proxies)
            self.candidate_label_map = candidate_label_map
            self.final_label_map = final_label_map
            self.initial_label_map = initial_label_map
            self.cand_sub_elm_vals = [final_label_map[elm] for elm in candidate_label_map if elm in final_label_map]
            self.concatenated = self.concat_init_cand()
            # print(self.concatenated)
        
    def l2_norm(self, x):
        input_size = x.size()
        buffer = torch.pow(x, 2)
        normp = torch.sum(buffer, 1).add_(1e-12)
        norm = torch.sqrt(normp)
        _output = torch.div(x, norm.view(-1, 1).expand_as(x))
        output = _output.view(input_size)
        return output
    
    def l2_norm_list(self, x):
        x_ = []
        for elm in x:
            x_.append(self.l2_norm(elm.unsqueeze(0)))
            # x_.append(elm.unsqueeze(0))
        return x_
    
    def concat_list(self,tensor_list):
        for cnt,elm in enumerate(tensor_list):
            if cnt == 0:
                out = elm
            else:
                out = torch.cat((out,elm), dim = 0)
        return out
    
    def concat_init_cand(self):
        concatenated = []
        for cnt,final_label in enumerate(self.final_label_map):
            if final_label in self.candidate_label_map:
                concatenated.append(self.candidate_proxies[self.candidate_label_map[final_label]])
            elif final_label in self.initial_label_map:
                concatenated.append(self.base_proxies[self.initial_label_map[final_label]])
        # concatenated = torch.stack(concatenated, dim=0)
        return concatenated
    
    def sim_func(self, debug = False, device = "cpu"):
        if self.task in ['add']:
            layers = self.l2_norm_list(self.concatenated)
            layers = self.concat_list(layers)
        elif self.task in ['create', 'enhance']:
            layers = self.l2_norm(self.base_proxies)
        
        sim_mat = torch.nn.functional.linear(layers.to(device), layers.to(device))
        similarity_vector = torch.triu(sim_mat,
                                       diagonal = 1)
        
        combinations_tuple = torch.nonzero(similarity_vector, as_tuple=True)
        combinations_list = torch.nonzero(similarity_vector, as_tuple=True)

        if self.task in ['add']:
            combinations = [elm for elm in combinations_list if (tuple(elm[0].cpu().numpy().tolist()) in self.cand_sub_elm_vals) or (elm[1] in self.cand_sub_elm_vals)]
            combinations = tuple([combinations[:,i] for i in range(combinations.shape[1])])
        elif self.task in ['create', 'enhance']:
            combinations = combinations_tuple
        
        similarity_vector = similarity_vector[combinations]
            
        del sim_mat
        torch.cuda.empty_cache()
        
        if debug:
            return similarity_vector, combinations
        else:
            return similarity_vector

def new_loss_func(similarity):
    # makes the angles between proxies maximum.
    loss = -torch.log(1-similarity)*1
    loss = torch.clamp(loss, min=0, max=8)
    return loss

def pop_run(POP,
            max_iter = 300, desired_sim_score = 0.08,
            loss_type = 'new', lr = 0.3,
            debug = False, verbose = True):
    
    # print("---> Defining proxies with Kaiming Normal Distribution")
    optimizer = torch.optim.Adam(POP.parameters(), lr=lr)
    lossfunc = torch.nn.MSELoss()
    POP.to("cpu")
    POP.train()
    
    print("---> Running Task {}".format(POP.task.upper()))
    print("---> Desired Similarity Score:", desired_sim_score)
    print("---> Max Iteration:", max_iter)
    cnt = 0
    loss_max = 1
    while loss_max>desired_sim_score:
        optimizer.zero_grad()
        distance_vector = POP.sim_func()
        
        if cnt == 0:
            init_max_loss = loss_max
        
        if loss_type == 'max':
            loss_max = torch.max(torch.abs(distance_vector-torch.zeros(distance_vector.shape)))
            loss_item = loss_max.item()
            loss_max.backward()
        elif loss_type == 'mean':
            loss_mean = torch.mean(torch.abs(distance_vector-torch.zeros(distance_vector.shape)))
            loss_item = loss_mean.item()
            loss_mean.backward()
        elif loss_type == 'mse':
            loss_mse = lossfunc(distance_vector, torch.zeros(distance_vector.shape))
            loss_item = loss_mse.item()
            loss_mse.backward()
        elif loss_type == "new":
            new_loss = new_loss_func(distance_vector).sum()
            loss_item = new_loss.item()
            new_loss.backward()
        angle = torch.rad2deg(torch.acos(torch.clip(distance_vector,
                                                    -0.9999,0.9999))).detach().numpy()
        if verbose:
            print('\nTASK: {} CNT: {}'.format(POP.task.upper(),cnt))
            print('Proxy {} LOSS: {} '.format(loss_type, loss_item))
            print('Min Angle: ', angle.min())
           
        optimizer.step()
        cnt += 1
        
        if cnt+1>=max_iter and angle.min()>90 and angle.max()<270:
            break
        
    del distance_vector
    torch.cuda.empty_cache()
    
    POP.eval()
    out = POP.base_proxies if POP.task in ['create', 'enhance'] else torch.cat((POP.base_proxies, POP.candidate_proxies), 0)
    if debug:
        return out.detach(), loss_max, init_max_loss
    else:
        return out.detach()

# import matplotlib.pyplot as plt
# import copy

# initial_label_map   = {'0': 0, '1': 1, '2': 2, '3': 3}
# candidate_label_map = {'4': 0, '5': 1}
# final_label_map = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5}

# base_proxies = torch.randn(4, 2)
# base_proxies_copy = copy.deepcopy(base_proxies)
# torch.nn.init.kaiming_normal_(base_proxies, mode='fan_out')

# pop_create = ProxyOperations(base_proxies = base_proxies,
#                              task = 'create')

# proxies, loss_max_create, init_max_loss_create = pop_run(pop_create, max_iter = 2000, desired_sim_score = 0.001, lr = 0.001, loss_type = 'max')

# plt.figure()
# base_proxies_copy = pop_create.l2_norm(base_proxies_copy)
# proxies = pop_create.l2_norm(proxies)
# plt.scatter(base_proxies_copy[:,0], base_proxies_copy[:,1], s = 300, c = 'b')
# plt.scatter(proxies[:,0], proxies[:,1], s = 50, c = 'r')
# plt.xlim([-1, 1])
# plt.ylim([-1, 1])
# plt.grid(which='major', alpha=1)

# # proxy addition
# candidate_proxies = torch.randn(2, 2)
# torch.nn.init.kaiming_normal_(candidate_proxies, mode='fan_out')
# candidate_proxies_copy = copy.deepcopy(candidate_proxies)

# pop_add = ProxyOperations(base_proxies = proxies,
#                           initial_label_map = initial_label_map,
#                           candidate_label_map = candidate_label_map,
#                           candidate_proxies = candidate_proxies,
#                           final_label_map = final_label_map,
#                           task = 'add')

# proxies_a, loss_max_add, init_max_loss_add = pop_run(pop_add, max_iter = 2000, desired_sim_score = 0.01, lr = 0.03, loss_type = 'max')

# plt.figure()
# candidate_proxies_copy = pop_add.l2_norm(candidate_proxies_copy)
# proxies_a = pop_add.l2_norm(proxies_a)
# plt.scatter(candidate_proxies_copy[:,0], candidate_proxies_copy[:,1], s = 300,  c = 'b')
# plt.scatter(proxies_a[:,0], proxies_a[:,1], s = 50,  c = 'r')
# plt.xlim([-1, 1])
# plt.ylim([-1, 1])
# plt.grid(which='major', alpha=0.1)

# proxies_a_copy = copy.deepcopy(proxies_a)

# pop_enhance = ProxyOperations(base_proxies = proxies_a,
#                               task = 'enhance')

# proxies_e, loss_max_enhance, init_max_loss_enhance = pop_run(pop_enhance, max_iter = 2000, desired_sim_score = 0.01, lr = 0.01, loss_type = 'max')

# plt.figure()
# proxies_a_copy = pop_add.l2_norm(proxies_a_copy)
# proxies_e = pop_add.l2_norm(proxies_e)
# plt.scatter(proxies_a_copy[:,0], proxies_a_copy[:,1], s = 300, c = 'b')
# plt.scatter(proxies_e[:,0], proxies_e[:,1], s = 50, c = 'r')
# plt.xlim([-1, 1])
# plt.ylim([-1, 1])
# plt.grid(which='major', alpha=0.1)