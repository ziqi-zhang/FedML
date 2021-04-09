import torch
from pdb import set_trace as st
import copy
import logging

class FeatAvgEnsemble(torch.nn.Module):
    def __init__(self, branches, client_list):
        super(FeatAvgEnsemble, self).__init__()
        model = client_list[0].model_trainer.model
        self.model_class = type(model)
        self.models = [copy.deepcopy(model) for _ in branches]
        self.training = False
        # self.models = [(client.model_trainer.model) for client in clients]
        # print(self.models[0].linear.weight.data_ptr() == self.models[1].linear.weight.data_ptr())
        ...
        
    def load_branch_to_models(self, branches, clients):
        for model, w in zip(self.models, branches):
            model.load_state_dict(w)


    def forward(self, x):
        with torch.no_grad():
            for block in self.model_class.blocks:
                feats, feat_cnt = None, 0
                for model in self.models:
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
        
        
