import torch
from pdb import set_trace as st

class PredVoteEnsemble(torch.nn.Module):
    def __init__(self, clients):
        super(PredVoteEnsemble, self).__init__()
        self.models = [client.model_trainer.model for client in clients]
        
    def update_clients(self, branches, clients):
        self.models = [client.model_trainer.model for client in clients]

    def forward(self, x):
        preds = [m(x) for m in self.models]
        preds_mean = torch.mean(torch.stack(preds), axis=0)

        preds = torch.stack([torch.max(pred, -1)[1] for pred in preds])
        preds = torch.mode(preds, 0)[0]
        return preds
    
    def to(self, device):
        for model in self.models:
            model.to(device)
            
    def eval(self):
        for model in self.models:
            model.eval()
