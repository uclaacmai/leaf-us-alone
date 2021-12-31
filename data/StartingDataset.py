import torch
import pandas as pd
import PIL.Image as img
import torchvision.transforms as t
import os

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, path, transform = None):
        df = pd.read_csv(path)
        self.imgid = df['image_id']
        self.labels = df['label']
        self.transform = transform
            

    def __getitem__(self, index):
        id = self.imgid.iloc[index]
        lbl = torch.tensor(int(self.labels.iloc[index]))

        return (t.ToTensor()(img.open(os.path.join('../train_images', str(id))).resize((224, 224))), lbl)

    def __len__(self):
        return len(self.labels)
