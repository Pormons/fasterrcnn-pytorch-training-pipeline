import torch

BATCH_SIZE = 1 # increase / decrease according to GPU memeory
RESIZE_TO = 416 # resize the image for training and transforms
NUM_EPOCHS = 25 # number of epochs to train for
NUM_WORKERS = 4

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# training images and XML files directory
TRAIN_DIR = 'data/Aquarium Combined.v2-raw-1024.voc/train'
# validation images and XML files directory
VALID_DIR = 'data/Aquarium Combined.v2-raw-1024.voc/valid'

# classes: 0 index is reserved for background
CLASSES = [
    '__background__', 'fish', 'jellyfish', 'penguin', 'shark', 
    'puffin', 'stingray', 'starfish'
]

NUM_CLASSES = len(CLASSES)

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = True

# location to save model and plots
OUT_DIR = 'outputs'