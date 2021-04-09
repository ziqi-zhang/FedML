import torch
from pdb import set_trace as st
import copy

class PredAvgEnsemble(torch.nn.Module):
    def __init__(self, clients):
        super(PredAvgEnsemble, self).__init__()
        self.models = [copy.deepcopy(client.model_trainer.model) for client in clients]
        # self.models = [(client.model_trainer.model) for client in clients]
        # print(self.models[0].linear.weight.data_ptr() == self.models[1].linear.weight.data_ptr())
        
    def update_clients(self, branches, clients):
        for model, w in zip(self.models, branches):
            model.load_state_dict(w)
        # for param1, param2 in zip(self.models[4].parameters(), self.models[8].parameters()):
        #     print((param1==param2).all())
        #     st()
        # st()

    def forward(self, x):
        preds = [m(x) for m in self.models]
        preds = torch.stack(preds)
        preds = torch.mean(preds, axis=0)
        return preds
    
    def to(self, device):
        for model in self.models:
            model.to(device)
            
    def eval(self):
        for model in self.models:
            model.eval()
