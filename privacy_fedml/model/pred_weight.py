import torch
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace as st
import logging
import copy

class PredWeight(torch.nn.Module):
    def __init__(self, branches, clients, args, output_dim=10):
        super(PredWeight, self).__init__()
        self.branches = branches
        self.models = nn.ModuleList([copy.deepcopy(client.model_trainer.model) for client in clients])
        self.branch_num = len(branches)
        self.output_dim = output_dim
        self.branch_weight = nn.Parameter(torch.ones(self.branch_num) / self.branch_num)
        # self.register_parameter("branch_weight", branch_weight)
        self.disable_module_grad()
        logging.info("*************New parameters")
        for name, param in self.named_parameters():
            logging.info(name)
        
        
    def update_clients(self, branches, clients):
        # for weight, client in zip(branches, clients):
        #     client.model_trainer.set_model_params(weight)
        #     print(client.model_trainer)
        # self.models = [client.model_trainer.model for client in clients]
        for model, w in zip(self.models, branches):
            model.load_state_dict(w)
        
    def disable_module_grad(self):
        self.model_grad_modules = {}
        for idx, model in enumerate(self.models):
            self.model_grad_modules[idx] = []
            for name, param in model.named_parameters():
                if param.requires_grad == True:
                    self.model_grad_modules[idx].append((name, param))
                    param.requires_grad = False
                    
            log = f"Model {idx} disable modules: "
            for (name, param) in self.model_grad_modules[idx]:
                log += f"{name}({param.data_ptr()}), "
            logging.info(log)
        
    def reset_module_grad(self):
        for idx, model in enumerate(self.models):
            log = f"Model {idx} reset modules: "
            for (name, param) in self.model_grad_modules[idx]:
                param.requires_grad = True
                log += f"{name}({param.data_ptr()}), "
            logging.info(log)
        
        

    def forward(self, x):
        # print(self.models[0].linear.weight.device)
        # st()
        preds = [m(x) for m in self.models]
        preds = torch.stack(preds)
        # logging.info(self.branch_weight)
        preds_pre = preds.clone()
        preds *= self.branch_weight.view(self.branch_num,1,1)
        assert preds[1,0,4] == preds_pre[1,0,4] * self.branch_weight[1]
        assert preds[6,4,3] == preds_pre[6,4,3] * self.branch_weight[6]
        
        preds = torch.sum(preds, axis=0)
        # preds = F.log_softmax(preds, dim=1)
        # logging.info(self.branch_weight)
        return preds
    
    # def to(self, device):
    #     for model in self.models:
    #         model.to(device)
    #     # self.branch_weight.data = self.branch_weight.to(device)
    #     print(self.branch_weight.device)
    #     st()
    # def eval(self):
    #     for model in self.models:
    #         model.eval()
