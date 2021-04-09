import torch
from torch import nn
from pdb import set_trace as st
import logging
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class MIAttackModelTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

        epoch_loss = []
        # if isinstance(model, PredWeight):
        #     st()
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                # logging.info("x.size = " + str(x.size()))
                # logging.info("labels.size = " + str(labels.size()))
                model.zero_grad()
                log_probs = model(x)
                
                loss = criterion(log_probs, labels)
                loss.backward()

                # to avoid nan loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # if isinstance(model, PredWeight):
                #     logging.info(model.branch_weight.grad)
                optimizer.step()
                # logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch, (batch_idx + 1) * self.args.batch_size, len(self.local_training_data) * self.args.batch_size,
                #            100. * (batch_idx + 1) / len(self.local_training_data), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info(f"MI attack model train epoch {epoch} loss {epoch_loss[-1]:.2f}")

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {}

        criterion = nn.CrossEntropyLoss().to(device)
        correct, total = 0, 0
        total_preds, total_targets = [], []
        
        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                if len(pred.shape) == 1:
                    predicted = pred
                    loss = torch.Tensor([0])

                else:
                    loss = criterion(pred, target)
                    _, predicted = torch.max(pred, -1)
                correct += predicted.eq(target).sum().item()
                total += target.size(0)
                total_preds.append(predicted.cpu().numpy())
                total_targets.append(target.cpu().numpy())
                
        metrics['acc'] = correct / total
        preds = np.concatenate(total_preds)
        targets = np.concatenate(total_targets)
        precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='macro')
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False


class MIAttackThred(ModelTrainer):
    def get_model_params(self):
        return None

    def set_model_params(self, model_parameters):
        ...

    def train(self, train_dataset, ):
        train_data = train_dataset.tensors[0].numpy()
        train_label = train_dataset.tensors[1].numpy()
        self.avg_loss = np.mean(train_data[train_label==1])
        # self.avg_loss = np.mean(train_data)
        

    def test(self, test_dataset, ):
        metrics = {}
        
        test_data = test_dataset.tensors[0].numpy()
        test_label = test_dataset.tensors[1].numpy()
        # test_pred = np.where(test_data <= self.avg_loss, 0, 1)
        test_pred = np.where(test_data <= self.avg_loss, 1, 0)
        # real_avg_loss = np.mean(test_data[test_label==0])

        correct = np.sum(test_label == test_pred)
        total = len(test_label)
                
        metrics['acc'] = correct / total
        precision, recall, f1, _ = precision_recall_fscore_support(test_label, test_pred, average='macro')
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False