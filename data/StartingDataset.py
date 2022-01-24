import torch, os
import constants
import pandas as pd
import PIL.Image as img
import PIL.ImageFilter as fil
import torchvision.transforms as t

# Determining how many of each image to use
TRAIN_NUM = constants.TRAIN_NUM
TEST_NUM = constants.TEST_NUM
TYPES = constants.IMG_TYPES
VALUES = constants.VALUES


class StartingDataset(torch.utils.data.Dataset):

    def __init__(self, path, transform = None, training_set = True):

        df = pd.read_csv(path)
        self.imgid = df['image_id']
        self.labels = df['label']

        self.transform = transform
        self.training_set = training_set
        self.dir = "/".join(path.split("/")[:-1])


        temp = []
        curr = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

        # First, augment images by adding three transformations
        for i in range(len(self.imgid)):
            finished = True
            for key in curr:
                if (curr[key] < TEST_NUM + TRAIN_NUM):
                    finished = False; break

            if finished: break
            if curr[self.labels[i]] >= TEST_NUM + TRAIN_NUM: continue

            if VALUES[self.labels[i]] >= TEST_NUM + TRAIN_NUM:
                # Means we do not need to augment the images (already have enough)
                temp.append([self.imgid.iloc[i], self.labels.iloc[i]])
                curr[self.labels[i]] += 1

            else:
                # Means we need to augment the images (will not have enough)
                temp_id = self.imgid.iloc[i].split('.jpg')[0]
                temp.append([temp_id + '.jpg', self.labels.iloc[i]])
                temp.append([temp_id + '_R.jpg', self.labels.iloc[i]])
                temp.append([temp_id + '_F.jpg', self.labels.iloc[i]])
                temp.append([temp_id + '_B.jpg', self.labels.iloc[i]])
                curr[self.labels[i]] += 4


        temp = pd.DataFrame(data=temp)

        self.imgid = temp[0]; self.labels = temp[1]

        self.imgid = self.imgid.sample(frac=1).reset_index(drop=True)
        self.labels = self.labels.sample(frac=1).reset_index(drop=True)

        # Next, specify which of the images are train and which are test
        if training_set:
            self.imgid = self.imgid[:TRAIN_NUM * TYPES]
            self.labels = self.labels[:TRAIN_NUM * TYPES]
        else:
            self.imgid = self.imgid[TRAIN_NUM * TYPES:]
            self.labels = self.labels[TRAIN_NUM * TYPES:]


        # Now, print some metrics just to make sure we have everything

        #mode = "training" if training_set else "testing"
        #print(f"Loaded up {len(self.imgid)} images for {mode}.")



    def __getitem__(self, index):
        id = self.imgid.iloc[index]
        lbl = torch.tensor(int(self.labels.iloc[index]))

        directory = self.dir + '/train_images'

        if '_' in id:
            image = img.open(os.path.join(directory, str(id[:id.find('_')] + '.jpg'))).resize((227, 227))
        else:
            image = img.open(os.path.join(directory, str(id))).resize((227, 227))

        if '_R' in id:
            image = image.rotate(180)
        elif '_F' in id:
            image = image.transpose(img.FLIP_LEFT_RIGHT)
        elif '_B' in id:
            image = image.filter(fil.BLUR)

        return (t.ToTensor()(image), lbl)

    def __len__(self):
        return len(self.labels)
