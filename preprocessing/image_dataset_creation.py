import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np

#change if necessary
PATH = '../data/'

normalize = transforms.Normalize(
   mean=[0.37204376, 0.31047195, 0.26713085],
   std=[0.00023457, 0.00027677, 0.00029105]
)

preprocess = transforms.Compose([
   transforms.Resize(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize
])

def video_loader(start, end, image_loader):
    """Helper function for loading video data"""
    frame_indx = np.linspace(start, end, num=32, dtype=int)
    return torch.stack([image_loader[i][0] for i in frame_indx])

#will be used to load images
class ImageDataset(Dataset):
    """Dataset combining visual, textual, auditory and microexpression features dataset."""

    def __init__(self, root_annotation, image_annotation,
                 root_dir_images, transform=None):
        """
        Args:
            root_annotation (string): Path to the csv file with annotations on guestures
            image_annotation (string): Directory with annotation relating to videos (start and end index in imageloader).
            root_dir_images: root directory for images
            transform (callable, optional): Optional transform to be applied
                on a sample images of video
        """
        self.annotation = pd.read_csv(root_annotation)
        self.image_annotation = pd.read_csv(image_annotation, index_col='name')
        self.images = torchvision.datasets.ImageFolder(root_dir_images, transform=preprocess)
        self.transform = transform


    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        vid_name = self.annotation.iloc[idx, 0]
        start = self.image_annotation.loc[vid_name[:-4], 'start']
        end = self.image_annotation.loc[vid_name[:-4], 'end']
        images = video_loader(start, end, self.images)
        return images


print('[INFO] Loading Dataset')
dataset = ImageDataset(root_annotation=PATH+'Real-life_Deception_Detection_2016/Annotation/All_Gestures_Deceptive and Truthful.csv',
                            image_annotation=PATH+'video_metadata.csv',
                            root_dir_images=PATH+'Frames/')

images_saved = []
for element in dataset:
    image = element.numpy()
    images_saved.append(image)
print('[INFO] Saving dataset')
images = np.array(images_saved)

np.save(PATH+'images_normalized', images)
