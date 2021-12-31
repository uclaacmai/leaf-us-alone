import argparse
import os
import torch

# lsfjalsdghlksghklghsaklghsadklghas fkldwgasglasgjslkfjsfklas

import constants
from data.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train import starting_train

train_dataset = StartingDataset('../train.csv')

print(train_dataset.__getitem__(index=1)[0][0][0][0], '\n')