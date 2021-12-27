import torch
import pandas as pd


class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, path):
        df = pd.read_csv(path)
        self.images = df['image_id']
        self.labels = df['label']

    def __getitem__(self, index):
        inputs = torch.zeros([3, 224, 224])
        label = 0

        return inputs, label

    def __len__(self):
        return len(self.labels)
