from src.datasets import MnistRotated, Paired_MnistRotated
from src.models import CNN

import wandb
import torch
import random
from tqdm.auto import tqdm

# ERM on RotatedMNIST
class ERM(object):
    def __init__(self, hparam):
        self.hparam = hparam
        self.device = self.hparam['device']
        self.train_set = MnistRotated(root=self.hparam["root"],
                 list_train_domains=self.hparam['training_angles'],
                 test_angle=self.hparam['test_angle'],
                 use_trainmnist_to_test=False,
                 train=True,
                 mnist_subset='max',
                 transform=None,
                 download=True)
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.hparam['batch_size'], shuffle=True)
        self.test_set = MnistRotated(root=self.hparam["root"],
                 list_train_domains=self.hparam['training_angles'],
                 test_angle=self.hparam['test_angle'],
                 use_trainmnist_to_test=False,
                 train=False,
                 mnist_subset='max',
                 transform=None,
                 download=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.hparam['batch_size'], shuffle=False)
        self.model = torch.nn.DataParallel(CNN(input_shape=(1,28,28)))
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparam['lr'])
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    def fit(self):
        print(len(self.train_set))
        for i in tqdm(range(self.hparam['epochs'])):
            self.model.train()
            total_loss = 0.
            for x,y_true,metadata in tqdm(self.train_loader):
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                metadata = metadata.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, y_true)
                with torch.no_grad():
                    total_loss += loss.item() * len(y_true)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if self.hparam['wandb']:
                wandb.log({"training_loss": total_loss / len(self.train_set)}, step=i)
            else:
                print(total_loss / len(self.train_set))
            self.evaluate(i)
    
    def evaluate(self, step):
        self.model.eval()
        corr = 0.
        for x,y_true,metadata in self.test_loader:
            x = x.to(self.device)
            y_true = y_true.to(self.device)
            metadata = metadata.to(self.device)
            outputs = self.model(x)
            y_pred = torch.argmax(outputs, dim=-1)
            corr += torch.sum(torch.eq(y_pred, y_true))
        if self.hparam['wandb']:
            wandb.log({"test_acc": corr / len(self.test_set)}, step=step)
        else:
            print(corr / len(self.test_set))
            

class Pair_Augmentation(object):
    def __init__(self, hparam):
        self.hparam = hparam
        self.device = self.hparam['device']
        self.train_set = Paired_MnistRotated(root=self.hparam["root"],
                 list_train_domains=['0','15','30','45','60'],
                 test_angle='75',
                 use_trainmnist_to_test=False,
                 train=True,
                 augmentation=self.hparam['augmentation'],
                 transform=None,
                 download=True)
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.hparam['batch_size'], shuffle=True)
        self.test_set = Paired_MnistRotated(root=self.hparam["root"],
                 list_train_domains=['0','15','30','45','60'],
                 test_angle='75',
                 use_trainmnist_to_test=False,
                 train=False,
                 augmentation=self.hparam['augmentation'],
                 transform=None,
                 download=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.hparam['batch_size'], shuffle=False)
        self.model = torch.nn.DataParallel(CNN(input_shape=(1,28,28)))
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparam['lr'])
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.criterion_2 = torch.nn.MSELoss(reduction='mean')
        self.training_angles = self.hparam['training_angles']
        self.test_angle = self.hparam['test_angle']

    def fit(self):
        for i in tqdm(range(self.hparam['epochs'])):
            self.model.train()
            total_loss = 0.
            total_celoss = 0.
            total_penalty = 0.
            for x_1, x_2,y_true, _,_ in tqdm(self.train_loader):
                x_1 = x_1.to(self.device)
                x_2 = x_2.to(self.device)
            
                y_true = y_true.to(self.device)
                outputs_1 = self.model(x_1)
                outputs_2 = self.model(x_2)
                penalty = self.criterion_2(outputs_1, outputs_2)
                loss = self.criterion((outputs_1 + outputs_2) / 2, y_true)
                # print(loss, penalty)
                obj = loss + self.hparam["param"] * penalty
                with torch.no_grad():
                    total_celoss += loss * len(y_true)
                    total_penalty += penalty * len(y_true)
                    total_loss += (loss.item() + penalty.item()) * len(y_true)

                obj.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if self.hparam['wandb']:
                wandb.log({"CELoss": total_celoss.item() / len(self.train_set), "penalty": total_penalty.item() / len(self.train_set), "training_loss": total_loss / len(self.train_set)}, step=i)
            else:
                print(total_loss / len(self.train_set))
            self.evaluate(i)


    def evaluate(self, step):
        self.model.eval()
        corr = 0.
        for x, y_true, _ in self.test_loader:
            x = x.to(self.device)
            y_true = y_true.to(self.device)
            outputs = self.model(x)
            y_pred = torch.argmax(outputs, dim=-1)
            corr += torch.sum(torch.eq(y_pred, y_true))
        if self.hparam['wandb']:
            wandb.log({"test_acc": corr / len(self.test_set)}, step=step)
        else:
            print(corr / len(self.test_set))