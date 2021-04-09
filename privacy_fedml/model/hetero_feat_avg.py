import torch
from pdb import set_trace as st
import copy
import logging
import torch.nn.functional as F

class HeteroFeatAvgEnsemble(torch.nn.Module):
    def __init__(self, hetero_archs, branches, client_list):
        super(HeteroFeatAvgEnsemble, self).__init__()
        model = client_list[0].model_trainer.model
        self.model_class = type(model)
        self.models = [copy.deepcopy(arch) for arch in hetero_archs]
        self.training = False
        # self.models = [(client.model_trainer.model) for client in clients]
        # print(self.models[0].linear.weight.data_ptr() == self.models[1].linear.weight.data_ptr())
        ...
        
    def load_branch_to_models(self, branches, clients):
        for model, w in zip(self.models, branches):
            model.load_state_dict(w)


    # def forward(self, x):
    #     with torch.no_grad():
    #         for block in self.model_class.blocks:
    #             feats, feat_cnt = None, 0
    #             for model in self.models:
    #                 feat = x
    #                 for layer_key in block:
    #                     feat = model.layer_to_forward_fn[layer_key](model, feat)
    #                 if feats is None:
    #                     feats = feat
    #                 else:
    #                     feats += feat
    #                 feat_cnt += 1
    #                 # feats.append(feat)
    #             # x = torch.mean(torch.stack(feats), dim=0)
    #             x = feats / feat_cnt
    #             # x = feats[0]
    #     return x
    

    def forward(self, x):
        preds = []
        with torch.no_grad():
            for model in self.models:
                _, pred = model.feature_forward(x)
                # pred = F.softmax(pred, dim =-1)
                preds.append(pred)
            
            votes = torch.stack([torch.max(pred, -1)[1] for pred in preds])
            votes = torch.mode(votes, 0)[0]
            return votes
            
            # preds = torch.stack(preds)
            # preds = F.softmax(preds, dim =-1)
            # preds = torch.mean(preds, axis=0)
            # return preds
    
    def to(self, device):
        for model in self.models:
            model.to(device)
        return self
            
    def eval(self):
        for model in self.models:
            model.eval()
        self.training = False
            
    def train(self):
        for model in self.models:
            model.train()
        self.training = True



class HeteroFeatAvgEnsembleDefense(torch.nn.Module):
    # def __init__(self, hetero_archs, branches, client_list):
    def __init__(self, original_ensemble_model, adv_ensemble_info):
        super(HeteroFeatAvgEnsembleDefense, self).__init__()
        self.model_class = original_ensemble_model.model_class
        self.models = copy.deepcopy(original_ensemble_model.models)
        del original_ensemble_model.models
        self.training = False
        self.adv_ensemble_info = {}
        for (block, branch_idx) in adv_ensemble_info[0].values():
            if branch_idx in self.adv_ensemble_info:
                self.adv_ensemble_info[branch_idx].append(block)
            else:
                self.adv_ensemble_info[branch_idx] = [block]
        for (block, branch_idx) in adv_ensemble_info[1].values():
            if branch_idx in self.adv_ensemble_info:
                self.adv_ensemble_info[branch_idx].append(block)
            else:
                self.adv_ensemble_info[branch_idx] = [block]

        
        # self.models = [(client.model_trainer.model) for client in clients]
        # print(self.models[0].linear.weight.data_ptr() == self.models[1].linear.weight.data_ptr())
        ...
        


    def forward(self, x):
        
        with torch.no_grad():
            for block in self.model_class.blocks:
                feats, feat_cnt = None, 0
                for branch_idx, model in enumerate(self.models):
                    # if branch_idx in self.adv_ensemble_info and block in self.adv_ensemble_info[branch_idx]:
                    if branch_idx in self.adv_ensemble_info:
                        continue
                    feat = x
                    for layer_key in block:
                        feat = model.layer_to_forward_fn[layer_key](model, feat)
                    if feats is None:
                        feats = feat
                    else:
                        feats += feat
                    feat_cnt += 1
                    # feats.append(feat)
                # x = torch.mean(torch.stack(feats), dim=0)
                x = feats / feat_cnt
                # x = feats[0]
        return x
    
    def to(self, device):
        for model in self.models:
            model.to(device)
        return self
            
    def eval(self):
        for model in self.models:
            model.eval()
        self.training = False
            
    def train(self):
        for model in self.models:
            model.train()
        self.training = True