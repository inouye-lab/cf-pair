import torch
import torch.autograd as autograd
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

from wilds.datasets.wilds_dataset import WILDSSubset
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.utils import split_into_groups
from wilds.common.grouper import CombinatorialGrouper

import wandb
from tqdm.auto import tqdm

from src.models import CNN, Classifier, ResNet18, ResNet50, ResNet101
from src.datasets import RotatedMNIST, ILDRotatedMNIST
from src.splitter import RandomSplitter
from src.ild_models import SimpleG, F_VAE_auto_soft_can, GDeep, GBetaVAE, F_VAE_auto_spa_can, F_VAE_auto_spa

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score


class ERM(object):
    def __init__(self, hparam):
        self.hparam = hparam
        self.device = self.hparam['device']
        self.dataset = RotatedMNIST(root_dir=self.hparam["root"], split_scheme=self.hparam["split_scheme"])
        # self.alter_dataset = get_dataset(dataset="celebA", root_dir=self.hparam["root"], download=True)
        # print(self.alter_dataset.metadata_array.shape)
        self.grouper = CombinatorialGrouper(dataset=self.dataset, groupby_fields=self.domain_fields)
        self.train_set = self.dataset.get_subset(split='train')
        self.train_loader = get_train_loader(self.loader_type, self.train_set, batch_size=self.hparam['batch_size'], uniform_over_groups=None, grouper=self.grouper, distinct_groups=True, n_groups_per_batch=self.n_groups_per_batch)
        self.test_set = self.dataset.get_subset(split='test')
        self.test_loader = get_eval_loader(loader='standard', dataset=self.test_set, batch_size=self.hparam["batch_size"])
        # self.featurizer = CNN(input_shape=(1,28,28), n_outputs=self.hparam['feature_dimension'])
        self.featurizer = ResNet18(input_shape=(1,28,28), n_outputs=self.hparam['feature_dimension'])
        self.classifier = Classifier(in_features=self.hparam['feature_dimension'], out_features=10, is_nonlinear=True)
        self.model = torch.nn.DataParallel(torch.nn.Sequential(self.featurizer, self.classifier))
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparam['lr'])
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    def fit(self):
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

    @property
    def loader_type(self):
        return 'standard'

    @property
    def domain_fields(self):
        return ['domain']

    @property
    def n_groups_per_batch(self):
        return 1


class CF_Pair(ERM):
    def fit(self):
        for i in tqdm(range(self.hparam['epochs'])):
            self.model.train()
            total_loss = 0.
            total_celoss = 0.
            total_penalty = 0.
            for x, y_true, metadata in tqdm(self.train_loader):
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                g = self.grouper.metadata_to_group(metadata).to(self.device)
                metadata = metadata.to(self.device)
                
                _, group_indices, _ = split_into_groups(g)
                group_indices = self.form_group(group_indices)
                features = self.featurizer(x)
                outputs = self.classifier(features)
                penalty = 0.
                features_0 = features[group_indices[0]]
                features_1 = features[group_indices[1]]
                penalty = self.distance(features_0, features_1)
                loss = self.criterion(outputs, y_true)
                # print(loss, penalty)
                obj = loss + self.hparam["param1"] * penalty
                with torch.no_grad():
                    total_celoss += loss * len(y_true)
                    total_penalty += penalty * len(y_true)
                    total_loss += (loss.item() + self.hparam["param1"] * penalty.item()) * len(y_true)
                obj.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if self.hparam['wandb']:
                wandb.log({"erm_loss": total_celoss.item() / len(self.train_set), "cf_loss": total_penalty.item() / len(self.train_set), "training_loss": total_loss / len(self.train_set)}, step=i)
            else:
                print(total_loss / len(self.train_set))
            self.evaluate(i)
    
    @property
    def loader_type(self):
        return 'group'

    @property
    def domain_fields(self):
        return ['id']
    
    @property
    def n_groups_per_batch(self):
        return int(self.hparam['batch_size'] / 2)
    
    @property
    def distance(self):
        return torch.nn.MSELoss(reduction='mean')

    def form_group(self, group_indices):
        return torch.stack(group_indices, axis=0).T

    
class Conditional_Pair(CF_Pair):
    def fit(self):
        for i in tqdm(range(self.hparam['epochs'])):
            self.model.train()
            total_loss = 0.
            total_celoss = 0.
            total_penalty = 0.
            for x, y_true, metadata in tqdm(self.train_loader):
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                g = self.grouper.metadata_to_group(metadata).to(self.device)
                metadata = metadata.to(self.device)
                _, group_indices, _ = split_into_groups(g)
                group_indices = self.form_group(group_indices)
                outputs = self.model(x)
                penalty = 0.
                output_0 = outputs[group_indices[0]]
                output_1 = outputs[group_indices[1]]
                penalty = self.distance(output_0, output_1)
                loss = self.criterion(output_0, y_true[group_indices[0]]) + self.criterion(output_1, y_true[group_indices[1]])
                # print(loss, penalty)
                obj = loss + self.hparam["param1"] * penalty
                with torch.no_grad():
                    total_celoss += loss * len(y_true)
                    total_penalty += penalty * len(y_true)
                    total_loss += (loss.item() + self.hparam["param1"] * penalty.item()) * len(y_true)

                obj.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if self.hparam['wandb']:
                wandb.log({"CELoss": total_celoss.item() / len(self.train_set), "penalty": total_penalty.item() / len(self.train_set), "training_loss": total_loss / len(self.train_set)}, step=i)
            else:
                print(total_loss / len(self.train_set))
            self.evaluate(i)
    
    @property
    def loader_type(self):
        return 'group'

    @property
    def domain_fields(self):
        return ['y']
    
    @property
    def n_groups_per_batch(self):
        return 2
    
    @property
    def distance(self):
        return torch.nn.MSELoss(reduction='mean')

    def form_group(self, group_indices):
        # debug
        group_indices = torch.stack(group_indices, axis=0).T
        n = group_indices.shape[0]
        return group_indices[0:n//2].flatten(), group_indices[n//2:].flatten()

    
class IRM(ERM):
    def __init__(self, hparam):
        super().__init__(hparam)
        self.scale = torch.tensor(1.).to(self.device).requires_grad_()
        self.update_count = 0
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.penalty_anneal_iters = self.hparam["param2"]
        self.penalty_weight = self.hparam["param1"]
        self.update_count = 0.
    
    def fit(self):
        for i in tqdm(range(self.hparam['epochs'])):
            self.model.train()
            total_loss = 0.
            total_celoss = 0.
            total_penalty = 0.
            for x, y_true, metadata in tqdm(self.train_loader):
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                g = self.grouper.metadata_to_group(metadata).to(self.device)
                metadata = metadata.to(self.device)
                
                _, group_indices, _ = split_into_groups(g)
                outputs = self.model(x)
                penalty = 0.
                loss = self.criterion(outputs*self.scale, y_true)
                penalty = self.irm_penalty(loss[group_indices[0]], loss[group_indices[1]])
                if self.update_count >= self.penalty_anneal_iters:
                    penalty_weight = self.penalty_weight
                else:
                    penalty_weight = self.update_count / self.penalty_anneal_iters
                avg_loss = loss.mean()
                obj = avg_loss + penalty_weight * penalty
                with torch.no_grad():
                    total_celoss += avg_loss * len(y_true)
                    total_penalty += penalty * len(y_true)
                    total_loss += (avg_loss.item() + self.hparam["param1"] * penalty.item()) * len(y_true)
                obj.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.update_count += 1

            if self.hparam['wandb']:
                wandb.log({"CELoss": total_celoss.item() / len(self.train_set), "penalty": total_penalty.item() / len(self.train_set), "training_loss": total_loss / len(self.train_set)}, step=i)
            else:
                print(total_loss / len(self.train_set))
            self.evaluate(i)


    def irm_penalty(self, loss_0, loss_1):
        grad_0 = autograd.grad(loss_0.mean(), [self.scale], create_graph=True)[0]
        grad_1 = autograd.grad(loss_1.mean(), [self.scale], create_graph=True)[0]
        result = torch.sum(grad_0 * grad_1)
        del grad_0, grad_1
        return result
    
    @property
    def loader_type(self):
        return 'group'

    @property
    def domain_fields(self):
        return ['domain']
    
    @property
    def n_groups_per_batch(self):
        return 2

    def form_group(self, group_indices):
        return group_indices
    

class CF_IRM(IRM):
    @property
    def domain_fields(self):
        return ['id']
    
    @property
    def n_groups_per_batch(self):
        return int(self.hparam['batch_size'] / 2)


    def form_group(self, group_indices):
        return torch.stack(group_indices, axis=0).T
    

class DeepCoral(ERM):
    def __init__(self, hparam):
        super().__init__(hparam)
        self.penalty_weight = self.hparam['param1']

    def fit(self):
        for i in tqdm(range(self.hparam['epochs'])):
            self.model.train()
            total_loss = 0.
            total_celoss = 0.
            total_penalty = 0.
            for x, y_true, metadata in tqdm(self.train_loader):
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                g = self.grouper.metadata_to_group(metadata).to(self.device)
                _, group_indices, _ = split_into_groups(g)
                metadata = metadata.to(self.device)

                features = self.featurizer(x)
                outputs = self.classifier(features)
                penalty = 0.
                loss = self.criterion(outputs, y_true)
                penalty = self.coral_penalty(features[group_indices[0]], features[group_indices[1]])
                
                obj = loss + self.penalty_weight * penalty
                with torch.no_grad():
                    total_celoss += loss * len(y_true)
                    total_penalty += penalty * len(y_true)
                    total_loss += (loss.item() + self.hparam["param1"] * penalty.item()) * len(y_true)
                obj.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            if self.hparam['wandb']:
                wandb.log({"CELoss": total_celoss.item() / len(self.train_set), "penalty": total_penalty.item() / len(self.train_set), "training_loss": total_loss / len(self.train_set)}, step=i)
            else:
                print(total_loss / len(self.train_set))
            self.evaluate(i)


    def coral_penalty(self, x, y):
        if x.dim() > 2:
            # featurizers output Tensors of size (batch_size, ..., feature dimensionality).
            # we flatten to Tensors of size (*, feature dimensionality)
            x = x.view(-1, x.size(-1))
            y = y.view(-1, y.size(-1))

        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff
    
    @property
    def domain_fields(self):
        return ['angle']
    
    @property
    def n_groups_per_batch(self):
        return 2

    def form_group(self, group_indices):
        return group_indices


class CF_DeepCoral(DeepCoral):
    @property
    def domain_fields(self):
        return ['id']
    
    @property
    def n_groups_per_batch(self):
        return int(self.hparam['batch_size'] / 2)

    def form_group(self, group_indices):
        return torch.stack(group_indices, axis=0).T


class Fewshot(ERM):
    """
    param 1: number of fine tune sample.
    param 2: penalty for the alignment
    param 3: penalty for the fine-tune set weight.
    """
    def __init__(self, hparam):
        self.hparam = hparam
        self.device = self.hparam['device']
        self.dataset = RotatedMNIST(root_dir=self.hparam["root"], split_scheme=self.hparam["split_scheme"])
        self.grouper = CombinatorialGrouper(dataset=self.dataset, groupby_fields=self.domain_fields)
        self.train_set = self.dataset.get_subset(split='train')
        idx = np.random.choice(60000, int(self.hparam['param1']), replace=False)
        finetune_idx = []
        for i in range(5):
            finetune_idx = np.concatenate([finetune_idx, idx + i * 60000])
        self.finetune_set = self.dataset[finetune_idx]
        print(type(self.finetune_set[0]))
        self.train_loader = get_train_loader('standard', self.train_set, batch_size=self.hparam['batch_size'], uniform_over_groups=None, grouper=self.grouper, distinct_groups=True, n_groups_per_batch=self.n_groups_per_batch)
        self.test_set = self.dataset.get_subset(split='test')
        self.test_loader = get_eval_loader(loader='standard', dataset=self.test_set, batch_size=self.hparam["batch_size"])
        self.featurizer = CNN(input_shape=(1,28,28), n_outputs=self.hparam['feature_dimension'])
        self.classifier = Classifier(in_features=self.hparam['feature_dimension'], out_features=10)
        self.model = torch.nn.DataParallel(torch.nn.Sequential(self.featurizer, self.classifier))
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparam['lr'])
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    def fit(self):
        for i in tqdm(range(self.hparam['epochs'])):
            total_loss = 0.
            erm_loss = 0.
            cf_loss = 0.
            self.model.train()
            for x, y_true,metadata in tqdm(self.train_loader):
                x = torch.cat((x, self.finetune_set[0])).to(self.device)
                y_true = torch.cat((y_true, self.finetune_set[1])).to(self.device)
                metadata = torch.cat((metadata, self.finetune_set[2])).to(self.device)

                features = self.featurizer(x)
                outputs = self.classifier(features)
                loss = self.criterion(outputs, y_true)
                feature_per_domain = features[-5*int(self.hparam['param1']):].reshape(5, int(self.hparam['param1']), int(self.hparam['feature_dimension']))
                feature_mean = feature_per_domain.mean(dim=0, keepdim=True)
                loss_2 = (torch.linalg.norm((feature_per_domain-feature_mean).flatten(), ord=2)  / self.hparam['param1']) ** 2
                obj = loss + self.hparam['param3'] * loss_2
                with torch.no_grad():
                    total_loss += obj.item() * len(y_true)
                    erm_loss += loss.item() * len(y_true)
                    cf_loss += loss_2.item() * len(y_true)
                self.optimizer.zero_grad()
                obj.backward()
                self.optimizer.step()
                
            if self.hparam['wandb']:
                wandb.log({"train_loss": total_loss / len(self.train_set), "erm_loss": erm_loss / len(self.train_set), "cf_loss": cf_loss / len(self.train_set)}, step=i)
            else:
                print({"train_loss": total_loss / len(self.train_set)})
            self.evaluate(i)
        self.optimizer.zero_grad()
    
    @property
    def loader_type(self):
        return 'group'

    @property
    def domain_fields(self):
        return ['id']
    
    @property
    def n_groups_per_batch(self):
        return int(self.hparam['batch_size'] / 2)
    
    @property
    def distance(self):
        return torch.nn.MSELoss(reduction='mean')

    def form_group(self, group_indices):
        return torch.stack(group_indices, axis=0).T
    

class EstimatedPair(ERM):
    def __init__(self, hparam):
        self.hparam = hparam
        self.device = self.hparam['device']
        self.dataset = ILDRotatedMNIST(root_dir=self.hparam["root"], split_scheme=self.hparam["split_scheme"], pair_path="saved_model/marginal_pair.npy")
        # self.alter_dataset = get_dataset(dataset="celebA", root_dir=self.hparam["root"], download=True)
        # print(self.alter_dataset.metadata_array.shape)
        self.grouper = CombinatorialGrouper(dataset=self.dataset, groupby_fields=self.domain_fields)
        self.train_set = self.dataset.get_subset(split='train')
        self.train_loader = get_train_loader(self.loader_type, self.train_set, batch_size=self.hparam['batch_size'], uniform_over_groups=None, grouper=self.grouper, distinct_groups=True, n_groups_per_batch=self.n_groups_per_batch)
        self.test_set = self.dataset.get_subset(split='test')
        self.test_loader = get_eval_loader(loader='standard', dataset=self.test_set, batch_size=self.hparam["batch_size"])
        self.featurizer = CNN(input_shape=(1,28,28), n_outputs=self.hparam['feature_dimension'])
        self.classifier = Classifier(in_features=self.hparam['feature_dimension'], out_features=10)
        self.model = torch.nn.DataParallel(torch.nn.Sequential(self.featurizer, self.classifier))
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparam['lr'])
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.pair = np.load(self.hparam['pair_path']).astype(int)

    def fit(self):
        for i in tqdm(range(self.hparam['epochs'])):
            self.model.train()
            total_loss = 0.
            total_celoss = 0.
            total_penalty = 0.
            for x, y_true, metadata, in tqdm(self.train_loader):
                idx = metadata[:, 4]
                pair_idx = self.pair[idx]
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                x_2, y_2, _ = self.dataset[pair_idx]
                x_2 = x_2.to(self.device)
                y_2 = y_2.to(self.device)

                feature_0 = self.featurizer(x)
                feature_1 = self.featurizer(x_2)
                output_0 = self.classifier(feature_0)
                output_1 = self.classifier(feature_1)
                penalty = 0.
                penalty = self.distance(feature_0, feature_1)
                loss = self.criterion(output_0, y_true) + self.criterion(output_1, y_2)
                # print(loss, penalty)
                obj = loss + self.hparam["param1"] * penalty
                with torch.no_grad():
                    total_celoss += loss * len(y_true)
                    total_penalty += penalty * len(y_true)
                    total_loss += (loss.item() + self.hparam["param1"] * penalty.item()) * len(y_true)
                obj.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if self.hparam['wandb']:
                wandb.log({"CELoss": total_celoss.item() / len(self.train_set), "penalty": total_penalty.item() / len(self.train_set), "training_loss": total_loss / len(self.train_set)}, step=i)
            else:
                print(total_loss / len(self.train_set))
            self.evaluate(i)
    
    @property
    def distance(self):
        return torch.nn.MSELoss(reduction='mean')
    

class CF_Contrastive(object):
    def __init__(self, hparam):
        self.hparam = hparam
        self.device = self.hparam['device']
        self.dataset = RotatedMNIST(root_dir=self.hparam["root"], split_scheme=self.hparam["split_scheme"])
        # self.alter_dataset = get_dataset(dataset="celebA", root_dir=self.hparam["root"], download=True)
        # print(self.alter_dataset.metadata_array.shape)
        self.grouper = CombinatorialGrouper(dataset=self.dataset, groupby_fields=['id'])
        self.train_set = self.dataset.get_subset(split='train')
        self.pair_loader = get_train_loader('group', self.train_set, batch_size=self.hparam['batch_size'], uniform_over_groups=None, grouper=self.grouper, distinct_groups=True, n_groups_per_batch=int(self.hparam['batch_size'] / 2))
        
        self.random_loader = get_train_loader('standard', self.train_set, batch_size=self.hparam['batch_size'])
        
        
        self.test_set = self.dataset.get_subset(split='test')
        self.test_loader = get_eval_loader(loader='standard', dataset=self.test_set, batch_size=self.hparam["batch_size"])
        self.featurizer = CNN(input_shape=(1,28,28), n_outputs=self.hparam['latent_dim'])
        self.classifier = Classifier(in_features=self.hparam['latent_dim'], out_features=10)
        self.model = torch.nn.DataParallel(torch.nn.Sequential(self.featurizer, self.classifier))
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparam['lr'])
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    def fit(self):
        for i in tqdm(range(self.hparam['epochs'])):
            self.model.train()
            total_loss = 0.
            total_celoss = 0.
            total_penalty = 0.
            for (x, y_true, metadata), (x_r, y_r, metadata_r) in tqdm(zip(self.pair_loader, self.random_loader)):
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                g = self.grouper.metadata_to_group(metadata).to(self.device)
                metadata = metadata.to(self.device)
                
                _, group_indices, _ = split_into_groups(g)
                group_indices = self.form_group(group_indices)
                features = self.featurizer(x)
                outputs = self.classifier(features)
                penalty = 0.
                features_0 = features[group_indices[0]]
                features_1 = features[group_indices[1]]
                penalty = self.distance(features_0, features_1)
                loss = self.criterion(outputs, y_true)
                # random_pair
                x_r = x_r.to(self.device)
                # y_r = y_r.to(self.device)
                feature_r = self.featurizer(x_r)
                penalty_r = self.distance(feature_r[0::2], feature_r[1::2])
                penalty_r = torch.clamp(penalty_r, max=self.hparam['param2'])
                obj = loss + self.hparam["param1"] * penalty - self.hparam["param3"] * penalty_r
                with torch.no_grad():
                    total_celoss += loss * len(y_true)
                    total_penalty += penalty * len(y_true)
                    total_loss += (loss.item() + self.hparam["param1"] * penalty.item()) * len(y_true)
                obj.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if self.hparam['wandb']:
                wandb.log({"CELoss": total_celoss.item() / len(self.train_set), "penalty": total_penalty.item() / len(self.train_set), "training_loss": total_loss / len(self.train_set)}, step=i)
            else:
                print(total_loss / len(self.train_set))
            self.evaluate(i)
        
    def form_group(self, group_indices):
        return torch.stack(group_indices, axis=0).T
    
    @property
    def distance(self):
        return torch.nn.MSELoss(reduction='mean')

    def form_group(self, group_indices):
        return torch.stack(group_indices, axis=0).T
    
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


class EstimatedContrastivePair(ERM):
    def __init__(self, hparam):
        self.hparam = hparam
        self.device = self.hparam['device']
        self.dataset = ILDRotatedMNIST(root_dir=self.hparam["root"], split_scheme=self.hparam["split_scheme"], pair_path="saved_model/marginal_pair.npy")
        # self.alter_dataset = get_dataset(dataset="celebA", root_dir=self.hparam["root"], download=True)
        # print(self.alter_dataset.metadata_array.shape)
        self.grouper = CombinatorialGrouper(dataset=self.dataset, groupby_fields=self.domain_fields)
        self.train_set = self.dataset.get_subset(split='train')
        self.train_loader = get_train_loader(self.loader_type, self.train_set, batch_size=self.hparam['batch_size'], uniform_over_groups=None, grouper=self.grouper, distinct_groups=True, n_groups_per_batch=self.n_groups_per_batch)
        self.test_set = self.dataset.get_subset(split='test')
        self.test_loader = get_eval_loader(loader='standard', dataset=self.test_set, batch_size=self.hparam["batch_size"])
        self.featurizer = CNN(input_shape=(1,28,28), n_outputs=self.hparam['latent_dim'])
        self.classifier = Classifier(in_features=self.hparam['latent_dim'], out_features=10)
        self.model = torch.nn.DataParallel(torch.nn.Sequential(self.featurizer, self.classifier))
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparam['lr'])
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.pair = np.load(self.hparam['pair_path']).astype(int)

    def fit(self):
        for i in tqdm(range(self.hparam['epochs'])):
            self.model.train()
            total_loss = 0.
            total_celoss = 0.
            total_pos_penalty = 0.
            total_neg_penalty = 0.
            for x, y_true, metadata, in tqdm(self.train_loader):
                idx = metadata[0::2, 4]
                pair_idx = self.pair[idx]
                x = x.to(self.device)
                y_true = y_true.to(self.device)
                x_2, y_2, _ = self.dataset[pair_idx]
                x_2 = x_2.to(self.device)
                y_2 = y_2.to(self.device)

                feature = self.featurizer(x)
                feature_0 = feature[0::2]
                feature_1 = feature[1::2] 
                feature_2 = self.featurizer(x_2)
                output = self.classifier(feature)
                output_2 = self.classifier(feature_2)
                penalty_pos = torch.sqrt(self.distance(feature_0, feature_2))
                penalty_neg = torch.sqrt(self.distance(feature_0, feature_1))
                loss = self.criterion(output, y_true) + self.criterion(output_2, y_2)
                # print(loss, penalty)
                obj = loss + self.hparam["param1"] * penalty_pos - self.hparam["param3"] * torch.clamp(penalty_neg, max=self.hparam['param2'])

                with torch.no_grad():
                    total_celoss += loss * len(y_true)
                    total_pos_penalty += penalty_pos * len(y_true)
                    total_neg_penalty += torch.clamp(penalty_neg, max=self.hparam['param2']) * len(y_true)
                    total_loss += obj.item() * len(y_true)
                obj.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            if self.hparam['wandb']:
                wandb.log({"CELoss": total_celoss.item() / len(self.train_set), "total_pos_penalty": total_pos_penalty.item() / len(self.train_set), "total_neg_penalty": total_neg_penalty.item() / len(self.train_set), "training_loss": total_loss / len(self.train_set)}, step=i)
            else:
                print(total_loss / len(self.train_set))
            self.evaluate(i)
    
    @property
    def distance(self):
        return torch.nn.MSELoss(reduction='mean')


# Contents
# 1. Constraint objective;
# 1.1. MSE. IRM, Fish, MMD.
# 2. Pairing Method;
# 2.1 Positive Pair
# 2.1.1 (Different Domain) Counterfactual Pair (dictionary lookup);
# 2.1.2 (Different Domain) Estimated Counterfactual Pair (dictionary lookup);
# 2.1.2.1 1-NN algorithm on ILD;
# 2.2.2.2 Sinkhorn-Barycenter algorithm on ILD;
# 2.2.2.3 1-NN algorithm on StarGAN;
# 2.2.2.4 Sinkhorn-Barycenter algorithm on StarGAN;
# 2.1.3 (Different Domain) Conditional Pair (Wilds Dataset);
# 2.2 Random Pair
# 2.2.1 (Different Domain) Random pair (Wilds Dataset);
# 2.2.2 Pure random pair (Subsample);
# 2.3 Negative Pair
# 2.3.1 Different Y pair?
# 3. Classification objective:
# 3.1 CE.

# First version: Pairing Constraint.
# class Pair(object):
#     # Based on Subsampling.
#     def __init__(self, hparam):
#         self.hparam = hparam
#         self.device = self.hparam['device']
#         self.dataset = RotatedMNIST(root_dir=self.hparam["root"], split_scheme=self.hparam["split_scheme"], pair_path="saved_model/marginal_pair.npy")
#         self.grouper = CombinatorialGrouper(dataset=self.dataset, groupby_fields=self.domain_fields)
#         self.train_set = self.dataset.get_subset(split='train')
#         self.train_loader = get_train_loader(self.loader_type, self.train_set, batch_size=self.hparam['batch_size'], uniform_over_groups=None, grouper=self.grouper, distinct_groups=True, n_groups_per_batch=self.n_groups_per_batch)
#         self.test_set = self.dataset.get_subset(split='test')
#         self.test_loader = get_eval_loader(loader='standard', dataset=self.test_set, batch_size=self.hparam["batch_size"])
#         # Feature, Classifier, Model.
#         self.featurizer = CNN(input_shape=(1,28,28), n_outputs=512)
#         self.classifier = Classifier(in_features=512, out_features=10, is_nonlinear=True)
#         self.model = torch.nn.DataParallel(torch.nn.Sequential(self.featurizer, self.classifier))
#         self.model.to(self.device)
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparam['lr'])
#         self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
#         self.pair_dict = np.load(self.hparam['pair_path']).astype(int)


#     def fit(self):
#         for i in tqdm(range(self.hparam['epochs'])):
#             self.model.train()
#             total_loss = 0.
#             total_celoss = 0.
#             total_penalty = 0.
#             for x, y_true, metadata, in tqdm(self.train_loader):
#                 idx = metadata[0::2, 4]
#                 pair_idx = self.pair[idx]
#                 x = x.to(self.device)
#                 y_true = y_true.to(self.device)
#                 x_2, y_2, _ = self.dataset[pair_idx]
#                 x_2 = x_2.to(self.device)
#                 y_2 = y_2.to(self.device)
#                 # feature = self.featurizer(x)
#                 # feature_0 = feature[0::2]
#                 # feature_1 = feature[1::2] 
#                 # feature_2 = self.featurizer(x_2)
#                 # output = self.classifier(feature)
#                 # output_2 = self.classifier(feature_2)
#                 # penalty_pos = torch.sqrt(self.distance(feature_0, feature_2))
#                 # penalty_neg = torch.sqrt(self.distance(feature_0, feature_1))
#                 # loss = self.criterion(output, y_true) + self.criterion(output_2, y_2)
#                 # print(loss, penalty)
#                 obj = loss + self.hparam["param1"] * torch.clamp(penalty_neg-penalty_pos, max=self.hparam['param2'])
#                 with torch.no_grad():
#                     total_celoss += loss * len(y_true)
#                     total_penalty += torch.clamp(penalty_neg-penalty_pos, max=self.hparam['param2']) * len(y_true)
#                     total_loss += obj.item() * len(y_true)
#                 obj.backward()
#                 self.optimizer.step()
#                 self.optimizer.zero_grad()
#             if self.hparam['wandb']:
#                 wandb.log({"CELoss": total_celoss.item() / len(self.train_set), "penalty": total_penalty.item() / len(self.train_set), "training_loss": total_loss / len(self.train_set)}, step=i)
#             else:
#                 print(total_loss / len(self.train_set)) 
#             self.evaluate(i)
    
#     @property
#     def penalty(self):
#         return torch.nn.MSELoss(reduction='mean')

#     # lookup
#     def pair(self, batch):
#         return batch[0::2], batch[1::2]


# class 

# # Contrastive: with negative pair
# class PairContrastive(object):
    
# # Triplet.
# class PairTriplet(object):

