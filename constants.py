"""
The following are constants used in training the model.
Kindly find a description of each as well as the default.

"""

# How many times should the model pass through the training set?
EPOCHS = 100

# How many images to use in one training iteration?
BATCH_SIZE = 30

# How often should we evaluate the model (in iterations)?
N_EVAL = 25

# How often should we save the model (in epochs, can be None)?
SAVE_INTERVAL = None 

# Where should we save the trained models to? 
SAVE_DIR = "saved_models"

# Where is the cassave-leaf-disease-classification dir?
DATA_DIR = '../cassava-leaf-disease-classification/'

# Where should we output our training summaries (in dir)?
SUMMARIES_PATH = "training_summaries"

LOG_DIR = "tb-outs"

# How many image types are there? (default 5)
IMG_TYPES = 5

# How many images of each type to use for training?
TRAIN_NUM = 2000

# How many images of each type to use for testing?
TEST_NUM = 200

# How many images of each type do we start with? (constant)
VALUES = {0: 1087, 1: 2189, 2: 2386, 3: 13158, 4: 2577}
