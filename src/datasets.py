"""Pytorch Dataset object that loads MNIST and SVHN. It returns x,y,s where s=0 when x,y is taken from MNIST."""

import os
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from itertools import permutations, combinations_with_replacement
import random


# code modified from https://github.com/atuannguyen/DIRT/blob/main/domain_gen_rotatedmnist/mnist_loader.py
class MnistRotated(data_utils.Dataset):
    def __init__(self,
                 root,
                 list_train_domains,
                 test_angle,
                 use_trainmnist_to_test=False,
                 train=True,
                 mnist_subset='med',
                 transform=None,
                 download=True):

        """
        :param list_train_domains: all domains we observe in the training
        :param root: data directory
        :param train: whether to load MNIST training data
        :param mnist_subset: 'max' - for each domain, use 60000 MNIST samples, 'med' - use 10000 MNIST samples, 'min' - use 1000 MNIST samples
        :param transform: ...
        :param download: ...
        :param list_test_domains: whether to load unseen domains (this might be removed later, but I don't have time to optimize the code at this point)
        :param num_supervised: whether to further subsample
        """
        self.root = os.path.expanduser(root)
        self.list_train_domains = list_train_domains
        self.test_angle = test_angle
        self.use_trainmnist_to_test = use_trainmnist_to_test
        self.train = train
        self.mnist_subset = mnist_subset
        self.transform = transform
        self.download = download

        # self.not_eval = not_eval  # load test MNIST dataset

        self.data, self.labels, self.domain, self.angles = self._get_data()

    def load_inds(self):
        '''
        If specifyign a subset, load 1000 mnist samples with balanced class (100 samples
        for each class). If not, load 10000 mnist samples.
        :return: indices of mnist samples to be loaded
        '''
        if self.mnist_subset=='med':
            fullidx = np.array([])
            for i in range(10):
                fullidx = np.concatenate(
                    (fullidx, np.load(os.path.join(self.root, 'roatedmnist_sup_inds/supervised_inds_' + str(i) + '.npy'))))
            return fullidx
        else:
            return np.load(os.path.join(self.root, 'roatedmnist_sup_inds/supervised_inds_' + self.mnist_subset + '.npy'))

    def _get_data(self):
        if self.train:
            train_loader = torch.utils.data.DataLoader(datasets.MNIST(self.root, train=True, download=self.download, transform=transforms.ToTensor()), batch_size=60000, shuffle=False)

            for i, (x, y) in enumerate(train_loader):
                mnist_imgs = x
                mnist_labels = y

            if self.mnist_subset != 'max':
                # Get labeled examples
                print(f'use MNIST subset {self.mnist_subset}!')
                sup_inds = self.load_inds()
                mnist_labels = mnist_labels[sup_inds]
                mnist_imgs = mnist_imgs[sup_inds]
            else:
                print('use all MNIST data!')

            
            self.num_supervised = int(mnist_imgs.shape[0])

            to_pil = transforms.ToPILImage()
            to_tensor = transforms.ToTensor()

            # Run transforms
            mnist_0_img = torch.zeros((self.num_supervised, 28, 28))
            mnist_15_img = torch.zeros((self.num_supervised, 28, 28))
            mnist_30_img = torch.zeros((self.num_supervised, 28, 28))
            mnist_45_img = torch.zeros((self.num_supervised, 28, 28))
            mnist_60_img = torch.zeros((self.num_supervised, 28, 28))
            mnist_75_img = torch.zeros((self.num_supervised, 28, 28))

            for i in range(len(mnist_imgs)):
                mnist_0_img[i] = to_tensor(to_pil(mnist_imgs[i]))

            for i in range(len(mnist_imgs)):
                mnist_15_img[i] = to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), 15))

            for i in range(len(mnist_imgs)):
                mnist_30_img[i] = to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), 30))

            for i in range(len(mnist_imgs)):
                mnist_45_img[i] = to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), 45))

            for i in range(len(mnist_imgs)):
                mnist_60_img[i] = to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), 60))

            for i in range(len(mnist_imgs)):
                mnist_75_img[i] = to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), 75))

            # Choose subsets that should be included into the training
            training_list_img = []
            training_list_labels = []
            train_angles = []
            for domain in self.list_train_domains:
                if domain == 0:
                    training_list_img.append(mnist_0_img)
                    training_list_labels.append(mnist_labels)
                    train_angles.append(0)
                if domain == 15:
                    training_list_img.append(mnist_15_img)
                    training_list_labels.append(mnist_labels)
                    train_angles.append(15) 
                if domain == 30:
                    training_list_img.append(mnist_30_img)
                    training_list_labels.append(mnist_labels)
                    train_angles.append(30) 
                if domain == 45:
                    training_list_img.append(mnist_45_img)
                    training_list_labels.append(mnist_labels)
                    train_angles.append(45) 
                if domain == 60:
                    training_list_img.append(mnist_60_img)
                    training_list_labels.append(mnist_labels)
                    train_angles.append(60) 
                if domain == 75:
                    training_list_img.append(mnist_75_img)
                    training_list_labels.append(mnist_labels)
                    train_angles.append(75) 

            # Stack
            train_imgs = torch.cat(training_list_img)
            train_labels = torch.cat(training_list_labels)

            # Create domain labels
            train_domains = torch.zeros(train_labels.size())
            train_domains[0: self.num_supervised] += 0
            train_domains[self.num_supervised: 2 * self.num_supervised] += 1
            train_domains[2 * self.num_supervised: 3 * self.num_supervised] += 2
            train_domains[3 * self.num_supervised: 4 * self.num_supervised] += 3
            train_domains[4 * self.num_supervised: 5 * self.num_supervised] += 4

            # Shuffle everything one more time
            inds = np.arange(train_labels.size()[0])
            np.random.shuffle(inds)
            train_imgs = train_imgs[inds]
            train_labels = train_labels[inds]
            train_domains = train_domains[inds].long()

            ## Convert to onehot
            #y = torch.eye(10)
            #train_labels = y[train_labels]

            ## Convert to onehot
            #d = torch.eye(5)
            #train_domains = d[train_domains]

            return train_imgs.unsqueeze(1), train_labels, train_domains, train_angles

        else:
            if self.use_trainmnist_to_test:
                bs = 60000
            else:
                bs = 10000
            train_loader = torch.utils.data.DataLoader(datasets.MNIST(self.root,
                                                                      train=self.use_trainmnist_to_test,
                                                                      download=self.download,
                                                                      transform=transforms.ToTensor()),
                                                       batch_size=bs,
                                                       shuffle=False)

            for i, (x, y) in enumerate(train_loader):
                mnist_imgs = x
                mnist_labels = y


            to_pil = transforms.ToPILImage()
            to_tensor = transforms.ToTensor()

            # Resize
            mnist_imgs_rot = torch.zeros((bs, 28, 28))
            for i in range(len(mnist_imgs)):
                mnist_imgs_rot[i] = to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), int(self.test_angle)))

            # Create domain labels
            test_domain = torch.zeros(mnist_labels.size()).long()

            return mnist_imgs_rot.unsqueeze(1), mnist_labels, test_domain, int(self.test_angle)

    def __len__(self):
        if self.train:
            return len(self.labels)
        else:
            if self.use_trainmnist_to_test:
                return 60000
            else:
                return 10000

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        d = self.domain[index]

        if self.transform is not None:
            x = self.transform(x)
        
        return x, y, d


class Paired_MnistRotated(data_utils.Dataset):
    def __init__(self,
                 root,
                 list_train_domains,
                 test_angle,
                 use_trainmnist_to_test=False,
                 train=True,
                 augmentation='cf',
                 transform=None,
                 download=True):

        """
        :param list_train_domains: all domains we observe in the training
        :param root: data directory
        :param train: whether to load MNIST training data
        :param mnist_subset: 'max' - for each domain, use 60000 MNIST samples, 'med' - use 10000 MNIST samples, 'min' - use 1000 MNIST samples
        :param transform: ...
        :param download: ...
        :param list_test_domains: whether to load unseen domains (this might be removed later, but I don't have time to optimize the code at this point)
        :param num_supervised: whether to further subsample
        """
        self.root = os.path.expanduser(root)
        self.list_train_domains = list_train_domains
        self.test_angle = test_angle
        self.use_trainmnist_to_test = use_trainmnist_to_test
        self.train = train
        self.augmentation = augmentation
        self.transform = transform
        self.download = download

        # self.not_eval = not_eval  # load test MNIST dataset
        self.data, self.labels, self.y_indices = self._get_data()

    
    def _get_data(self):
        if self.train:
            train_loader = torch.utils.data.DataLoader(datasets.MNIST(self.root, train=True, download=self.download, transform=transforms.ToTensor()), batch_size=60000, shuffle=False)

            for i, (x, y) in enumerate(train_loader):
                mnist_imgs = x
                mnist_labels = y

            y_idx = []
            for i in range(10):
                y_idx.append(((mnist_labels == i).nonzero(as_tuple=True)[0]).tolist())
            
            self.num_supervised = int(mnist_imgs.shape[0])

            to_pil = transforms.ToPILImage()
            to_tensor = transforms.ToTensor()

            # Run transforms
            mnist_img = torch.zeros((len(self.list_train_domains), self.num_supervised, 28, 28))
            
            for i, angle in enumerate(self.list_train_domains):
                for j in range(len(mnist_imgs)):
                    if angle == 0 or angle == '0':
                        mnist_img[i][j] = to_tensor(to_pil(mnist_imgs[j]))
                    else:
                        mnist_img[i][j] = to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[j]), int(angle)))




            return mnist_img.unsqueeze(2), mnist_labels, y_idx
        
        else:
            if self.use_trainmnist_to_test:
                bs = 60000
            else:
                bs = 10000
            train_loader = torch.utils.data.DataLoader(datasets.MNIST(self.root, train=self.use_trainmnist_to_test, download=self.download, transform=transforms.ToTensor()), batch_size=bs, shuffle=False)

            for i, (x, y) in enumerate(train_loader):
                mnist_imgs = x
                mnist_labels = y
            

            to_pil = transforms.ToPILImage()
            to_tensor = transforms.ToTensor()

            # Resize
            mnist_imgs_rot = torch.zeros((1, bs, 28, 28))
            for i in range(len(mnist_imgs)):
                mnist_imgs_rot[0][i] = to_tensor(transforms.functional.rotate(to_pil(mnist_imgs[i]), int(self.test_angle)))


            return mnist_imgs_rot.unsqueeze(2), mnist_labels, None

    def __len__(self):
        if self.train:
            return 60000
        else:
            if self.use_trainmnist_to_test:
                return 60000
            else:
                return 10000

    def __getitem__(self, index):
        if self.train == True and self.augmentation == "cf":
            sampled_domains = random.sample(list(range(len(self.list_train_domains))), 2)
            x_1 = self.data[sampled_domains[0]][index]
            x_2 = self.data[sampled_domains[1]][index]
            y = self.labels[index]
            d_1 = self.list_train_domains[sampled_domains[0]]
            d_2 = self.list_train_domains[sampled_domains[1]]
            return x_1, x_2, y, d_1, d_2
        elif self.train == True and self.augmentation == "unpaired":
            sampled_domains = random.sample(list(range(len(self.list_train_domains))), 2)
            x_1 = self.data[sampled_domains[0]][index]
            y = self.labels[index]
            sample_pair_idx = random.sample(self.y_indices[y.item()], 1)
            x_2 = self.data[sampled_domains[1]][sample_pair_idx[0]]
            
            d_1 = self.list_train_domains[sampled_domains[0]]
            d_2 = self.list_train_domains[sampled_domains[1]]
            return x_1, x_2, y, d_1, d_2
        elif self.train == True:
            raise NotImplementedError('the augmentation type is not supported yet.')
        else:
            x = self.data[0][index]
            y = self.labels[index]
            d = self.test_angle
        return x, y, d