from .proxy_anchor import ProxyAnchorLoss as ProxyVector
from .others import Proxy_NCA, MultiSimilarityLoss, ContrastiveLoss, \
    TripletLoss, NPairLoss, LinearProjection

import os
import torch
import copy
import tqdm

counter_device = "cpu"

def save_proxies(cfg, filename, proxies, label_map):
    try:
        os.mkdir('./proxies/model_{}'.format(cfg.dataset))
    except:
        pass
    try:
        os.mkdir('{}'.format('./proxies'+cfg.resume))
    except:
        pass
    data = {'proxies': proxies, 'label_map': label_map}
    torch.save(data, cfg.resume.replace("ckpt","{}.pth".format(filename)))

def load(cfg, model, trn_dataset, gen_dataset):

    if cfg.loss == 'ProxyAnchor':
        filepath = cfg.resume.replace("ckpt","{}.pth".format('proxies'))
        print("Proxies exist:", os.path.isfile(filepath), filepath)
        if os.path.isfile(filepath):
            print('=> loading proxies:\n{}'.format(filepath))
            data = torch.load(filepath)
            
            initial_proxies = data['proxies']
            initial_label_map = data['label_map']
            final_label_map = trn_dataset.label_map
            final_labels = sorted([elm for elm in final_label_map if elm not in initial_label_map] + [elm for elm in initial_label_map if elm not in final_label_map] + [elm for elm in initial_label_map if elm in final_label_map])
            print("After // Final Labels Length:", len(final_labels))
            final_label_map = {}
            for cnt,elm in enumerate(final_labels):
                final_label_map[elm] = cnt
                     
            if not (sorted(list(final_label_map.keys()))==sorted(list(initial_label_map.keys()))):
                old_classes = list(initial_label_map.keys())
                new_classes = [elm for elm in list(final_label_map.keys()) if elm not in old_classes]
        
                nb_classes = len(old_classes) + len(new_classes)
        
                #create candidate proxies
                candprox_dataset = copy.deepcopy(gen_dataset)
                dl_cand = torch.utils.data.DataLoader(candprox_dataset,
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
                            input_ = batch[0].to(cfg.device)
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
                                        sz_embed = cfg.embedding_size,
                                        mrg = cfg.mrg,
                                        alpha = cfg.alpha,
                                        pop_task = 'add',
                                        initial_proxies = initial_proxies,
                                        initial_label_map = initial_label_map,
                                        candidate_proxies = candidate_proxies,
                                        candidate_label_map = candidate_label_map,
                                        final_label_map = final_label_map)
                proxies = criterion.proxies
                save_proxies(cfg, 'vector_proxies', proxies, final_label_map)
            else:
                criterion = ProxyVector(nb_classes = trn_dataset.nb_classes(),
                                        sz_embed = int(cfg.embedding_size),
                                        mrg = cfg.mrg,
                                        alpha = cfg.alpha,
                                        pop_task = 'create',
                                        initial_proxies = initial_proxies,
                                        decompose = False)
        else:
            
            if cfg.decomposition:
                #create candidate proxies
                candprox_dataset = copy.deepcopy(gen_dataset)
                dl_cand = torch.utils.data.DataLoader(candprox_dataset,
                                                      batch_size = 1,
                                                      shuffle = False,
                                                      num_workers = cfg.num_workers,
                                                      pin_memory = True)
                
                model.eval()
                candidate_proxies_dict = {}
                with torch.no_grad():
                    print("Generating embedding...")
                    pbar = tqdm.tqdm(enumerate(dl_cand))
                    for batch_id, batch in pbar:
                        input_ = batch["image"].to(cfg.device)
                        out_ = model(input_).to(counter_device)
                        if batch["label_int"].item() in candidate_proxies_dict:
                            candidate_proxies_dict[batch["label_int"].item()] = torch.cat((candidate_proxies_dict[batch["label_int"].item()], 
                                                                                    out_), axis = 0)
                        else:
                            candidate_proxies_dict[batch["label_int"].item()] = out_
                        pbar.set_description("[{}/{} ({:.0f}%)]".format(
                                             batch_id + 1, len(dl_cand),
                                             100. * (batch_id+1) / len(dl_cand)))
                
                del batch
                del dl_cand
                del candprox_dataset
                
                candidate_label_map = {}
                for cnt,key in enumerate(candidate_proxies_dict):
                    candidate_label_map[key] = cnt
                    if cnt == 0:
                        out = candidate_proxies_dict[key].mean(dim=0).unsqueeze(0)
                    else:
                        out = torch.cat((out,
                                         candidate_proxies_dict[key].mean(dim=0).unsqueeze(0)), dim=0)
                candidate_proxies = torch.tensor(out, requires_grad = True)
            else:
                candidate_proxies = None
            final_label_map = trn_dataset.label_map
            nb_classes = len(final_label_map)
            criterion = ProxyVector(nb_classes = trn_dataset.nb_classes(),
                                    sz_embed = int(cfg.embedding_size),
                                    mrg = cfg.mrg,
                                    alpha = cfg.alpha,
                                    pop_task = 'create',
                                    initial_proxies = candidate_proxies,
                                    decompose = cfg.decomposition)
            proxies = criterion.proxies
            save_proxies(cfg, 'proxies', proxies, final_label_map)
            torch.cuda.empty_cache()
            
    elif cfg.loss == 'ProxyNCA':
        criterion = Proxy_NCA(nb_classes = trn_dataset.nb_classes(), sz_embed = cfg.embedding_size).to(counter_device)
    elif cfg.loss == 'MS':
        criterion = MultiSimilarityLoss().to(counter_device)
    elif cfg.loss == 'Contrastive':
        criterion = ContrastiveLoss().to(counter_device)
    elif cfg.loss == 'Triplet':
        criterion = TripletLoss().to(counter_device)
    elif cfg.loss == 'NPair':
        criterion = NPairLoss().to(counter_device)
    elif cfg.loss == "LinearProjection":
        criterion = LinearProjection(nb_classes = trn_dataset.nb_classes(),
                                     split_coefficient = cfg.split_coefficient).to(counter_device)
    else:
        raise ValueError('Unsupported loss function,\
                         must be one of ProxyAnchor, ProxyNCA, \
                         MS, Contrastive, Triplet, NPair')
                         
    return criterion