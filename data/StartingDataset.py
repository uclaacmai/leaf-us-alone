import torch
import pandas as pd
import PIL.Image as img

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, path):
        df = pd.read_csv(path)
        self.imgid = df['image_id']
        self.labels = df['label']
            

    def __getitem__(self, index):
        id = self.imgid.iloc(index)
        lbl = self.labels.iloc(index)

        return img.open('../../train_images/' + str(id) + '.jpg'), self.labels[index]

    def __len__(self):
        return len(self.labels)
