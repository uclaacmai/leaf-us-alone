import argparse
import os
import torch
from torchvision import transforms as t

# lsfjalsdghlksghklghsaklghsadklghas fkldwgasglasgjslkfjsfklas

import constants
from data.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train import starting_train

train_dataset = StartingDataset('../train.csv')

print(t.ToPILImage()(train_dataset[1][0]).show(), '\n')