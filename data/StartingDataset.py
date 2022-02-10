import torch, os
import constants
import pandas as pd
import PIL.Image as img
import PIL.ImageFilter as fil
import torchvision.transforms as t

# Determining how many of each image to use
TRAIN_NUM = constants.TRAIN_NUM # The number of images that will be used in training dataset. (int)
TEST_NUM = constants.TEST_NUM # The number of images that will be used in the test dtataset. (int)
TYPES = constants.IMG_TYPES # The number of different types of disease classifications in the dataset. (int)
VALUES = constants.VALUES # The counts of how many images that we start off with (dict)


class StartingDataset(torch.utils.data.Dataset):

    def __init__(self, path, training_set = True):

        df = pd.read_csv(path) # Convert the .csv file containing all the dataset info
                               # into a pandas dataframe.

        self.imgid = df['image_id'] # Intiailize a private dataframe 'imgid' to store the image ids for all of the
                                    # input images.

        self.labels = df['label'] # Intiailize a private dataframe 'label' to store the image classifications for all of the
                                  # input images.

        self.training_set = training_set # Boolean denoting whether or not we would like to use the traning set for this
                                         # dataset (if false, will use the validation set instead)

        self.dir = "/".join(path.split("/")[:-1])

        temp = []
        curr = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0} # Dictionary storing the current amount of values for each leaf classification.
                                              # This will get continuously updated throughout the following for-loop, as we increase
                                              # the number of images in each classification class.

        # First, augment images by adding three transformations

        for i in range(len(self.imgid)):
            finished = True # Bool value denoting if we're done creating new image types.

            # The following loop checks to see whether or not we have "enough" images
            # within each classification type, which indicates when we should stop
            # creating new images. The threshold for how many images is needed is determined
            # by the sumation of the TEST_NUM and TRAIN_NUM constants in the "constants.py" file.
            for key in curr:
                if (curr[key] < TEST_NUM + TRAIN_NUM):
                    finished = False; break
                        

            if finished: break #End the loop if finished.


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
                            # Grab the current image, create 3 more labels for the image
                            # to denote the augmentation type that will be done on it when being returned by the
                            # "__getitem__" function.


        temp = pd.DataFrame(data=temp) # Create a temporary dataframe for storing the newly created imageids
                                       # and the their corresponding values.

        temp = temp.sample(frac=1).reset_index(drop=True) # Randomize the images, and reset their indices.
        self.imgid = temp[0]; self.labels = temp[1] # Separate the images into their imgids and labels.

        # Next, specify which of the images are train and which are test
        if training_set:
            self.imgid = self.imgid[:TRAIN_NUM * TYPES]
            self.labels = self.labels[:TRAIN_NUM * TYPES]
        else:
            self.imgid = self.imgid[TRAIN_NUM * TYPES:]
            self.labels = self.labels[TRAIN_NUM * TYPES:]




    def __getitem__(self, index):
        id = self.imgid.iloc[index] # Grab the imageid at the current index.
        lbl = torch.tensor(int(self.labels.iloc[index])) # Grab the image label of the current
                            # image (need to convert it into a tensor in order to use it later on)

        directory = self.dir + '/train_images' # Grab the images.

        if '_' in id: # If the image has an associated agumentation, then we need to grab the OG image first before doing anything else.
            image = img.open(os.path.join(directory, str(id[:id.find('_')] + '.jpg'))).resize((227, 227))
        else: # If the image is not augmented, then it becomes trival to grab the image directly from the directory.
            image = img.open(os.path.join(directory, str(id))).resize((227, 227))

                    # Perform the necessary augmentations to the image.
        if '_R' in id:
            image = image.rotate(180)
        elif '_F' in id:
            image = image.transpose(img.FLIP_LEFT_RIGHT)
        elif '_B' in id:
            image = image.filter(fil.BLUR)

        return (t.ToTensor()(image), lbl)

    def __len__(self):
        return len(self.labels) # Return the length of the dataset (which is just the length of the dataframe).
