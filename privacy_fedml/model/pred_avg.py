import torch
from pdb import set_trace as st

class PredAvgEnsemble(torch.nn.Module):
    def __init__(self, clients):
        super(PredAvgEnsemble, self).__init__()
        self.models = [client.model_trainer.model for client in clients]
        
    def update_clients(self, clients):
        self.models = [client.model_trainer.model for client in clients]

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
