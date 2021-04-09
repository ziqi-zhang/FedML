import torch
from torch import nn
from pdb import set_trace as st
import logging
import copy
from itertools import chain

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer
from .model.pred_weight import PredWeight


class OneModelTrainer():
    def __init__(self, model, args=None):
        self.model = model
        
        self.id = 0
        self.args = args
        
    def weight_reinit(self):
        self.model.weight_reinit()

    def set_id(self, trainer_id):
        self.id = trainer_id
        
    def get_model_params(self):
        return (self.model.cpu().state_dict(), )
        
    def get_model_arch_weights(self):
        # return self.model1.cpu(), self.model2.cpu()
        return (self.model.cpu().state_dict(), )

    def set_model_params(self, params):
        self.model.load_state_dict(params)
        # self.model1.load_state_dict(params)
        
    def set_model_arch_weights(self, models):
        self.model = models[0]
        
    def model_weight_init(self, model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, GroupNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        for m in model.modules():
            if isinstance(m, Bottleneck):
                m.bn3.weight.data.fill_(0)
            if isinstance(m, BasicBlock):
                m.bn2.weight.data.fill_(0)
        

    def train(self, train_data, device, args):
        model = self.model
        # print(id(self.model1))
        # st()

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        featloss = torch.nn.MSELoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()), 
                lr=args.lr)
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), 
                lr=args.lr,
                weight_decay=args.wd, amsgrad=True)
        # if args.client_optimizer == "sgd":
        #     optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        # else:
        #     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
        #                                  weight_decay=args.wd, amsgrad=True)

        epoch_loss, feat_loss, cls_loss = [], [], []
        # if isinstance(model, PredWeight):
        #     st()
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                # logging.info("x.size = " + str(x.size()))
                # logging.info("labels.size = " + str(labels.size()))
                model.zero_grad()
                
                features, logit = model.feature_forward(x)
                loss = criterion(logit, labels)
                
                
                
                loss.backward()
                # loss2.backward()

                # to avoid nan loss
                # print(id(self.model))
                # st()
                # for p in self.model.parameters():
                #     if p.requires_grad:
                #         print(p.grad.shape)
                # st()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # if isinstance(model, PredWeight):
                #     logging.info(model.branch_weight.grad)
                optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
                #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
                # logging.info(
                #     f"Epoch {epoch} [{(batch_idx + 1) * args.batch_size}/{len(train_data) * args.batch_size}], "
                #     f"clsLoss {cls_loss.item():.3f}, featLoss {regloss.item():.3f}, totalLoss {loss.item():.3f}"
                    
                #     )
                batch_loss.append(loss.item())
                # break
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
            #     self.client_idx, epoch, sum(epoch_loss) / len(epoch_loss)))

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                
                features, logit = model.feature_forward(x)
                pred = logit
                loss = criterion(pred, target)
                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
                # break
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

# class TwoModelWarpper(nn.Module):
#     def __init__(self, model1, model2):
#         super(TwoModelWarpper, self).__init__()
#         self.model1, self.model2 = model1, model2
#         self.training = False
        
#     def to(self, device):
#         self.model1.to(device)
#         self.model2.to(device)
#         return self
        
#     def eval(self,):
#         self.model1.eval()
#         self.model2.eval()
        
#     def forward(self, x):
#         logit1 = self.model1(x)
#         logit2 = self.model2(x)
#         pred = (logit1 + logit2) / 2
#         return pred
