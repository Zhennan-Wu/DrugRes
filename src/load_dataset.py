import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os

class MNIST:
	def __init__(self, DATASET_DIR='../dataset/MNIST/'):
		self.DATASET_DIR = DATASET_DIR

	def fit_normalizer(self, x):
		self.min = np.min(x)
		self.max = np.max(x)

	def binarize(self, x):
		return (x > 0.5).astype(np.float32)
	
	def transform_normalizer(self, x):
		return (x - self.min)/(self.max - self.min)

	def inv_transform_normalizer(self, x):
		return (x * (self.max - self.min)) + self.min

	def load_dataset(self):
		test = pd.read_csv(self.DATASET_DIR+'test.csv')
		test = test.values
		train = pd.read_csv(self.DATASET_DIR+'train.csv')
		train = train.values
		test_x = test.T[1:].T
		test_y = test.T[0]
		train_x = train.T[1:].T
		train_y = train.T[0]

		train_x, test_x = train_x.astype(np.float32), test_x.astype(np.float32)
		self.fit_normalizer(train_x)
		train_x = self.transform_normalizer(train_x)
		test_x = self.transform_normalizer(test_x)
		train_x = self.binarize(train_x)
		test_x = self.binarize(test_x)

		train_x, train_y, test_x, test_y = torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(test_x), torch.from_numpy(test_y)	

		return train_x, train_y, test_x, test_y
	
class CIFAR10:
	def __init__(self, DATASET_DIR='../dataset/CIFAR10/'):
		self.DATASET_DIR = DATASET_DIR
		self.transform = transforms.Compose(
			[transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

		self.batch_size = 60000

		self.classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	def load_dataset(self):

		trainset = torchvision.datasets.CIFAR10(root=self.DATASET_DIR, train=True,
												download=True, transform=self.transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
												shuffle=True, num_workers=2)

		testset = torchvision.datasets.CIFAR10(root=self.DATASET_DIR, train=False,
											download=True, transform=self.transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
												shuffle=False, num_workers=2)
		
		train_x = None
		train_y = None
		for data in trainloader:
			images, labels = data
			train_x = images.reshape(images.shape[0], -1)
			train_y = labels

		test_x = None
		test_y = None
		for data in testloader:
			images, labels = data
			test_x = images.reshape(images.shape[0], -1)
			test_y = labels

		return train_x, train_y, test_x, test_y