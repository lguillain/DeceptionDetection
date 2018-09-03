import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np


class FastMultimodalDataset(Dataset):
    """Dataset combining visual, textual, auditory and microexpression features dataset."""

    def __init__(self, root_annotation,
                 root_images, root_audio, root_text,
                 pretrained_audio=lambda x: x, transform=None):
        """
        Args:
            root_annotation (string): Path to the csv file with annotations on guestures
            root_images: root directory for images
            root_audio: root directory containing data extracted from audio of videos
            root_text: transformed vectors from text
            transform (callable, optional): Optional transform to be applied
                on a sample images of video
        """
        self.annotation = pd.read_csv(root_annotation)
        self.images = np.load(root_images)
        self.audio = pd.read_pickle(root_audio)
        self.text = pd.read_pickle(root_text)
        self.pretrained_audio = pretrained_audio
        self.transform = transform


    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        vid_name = self.annotation.iloc[idx, 0]
        images = self.images[idx]

        text = self.text.loc[vid_name, 'Embedding']
        MAX_LEN_TEXT = 258#182
        text = np.append(text, np.zeros(MAX_LEN_TEXT - text.shape[0]), axis=0)
        #text = np.append(text, np.zeros((MAX_LEN_TEXT - text.shape[0], 300)), axis=0)

        audio = self.pretrained_audio(self.audio.loc[vid_name, 'Audio'])

        micro = self.annotation.iloc[idx, 1:-1].values.astype('float')

        self.sample = { 'audio': audio, 'microexpressions': micro,
        'text': text,
        'images': images}

        target = 0
        if self.text.loc[vid_name, 'Label'] == 'Truthful':
            target = 1

        if self.transform:
            sample = self.transform(sample)

        return self.sample, target


####  OLD CODE, MAY STILL BE OF USE FOR SAMPLE TESTING
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
   transforms.Resize(256),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   #normalize
])


def video_loader(start, end, image_loader):
    """Helper function for loading video data"""
    frame_indx = np.linspace(start, end, num=32, dtype=int)
    return torch.stack([image_loader[i][0] for i in frame_indx])


class MultimodalDataset(Dataset):
    """Dataset combining visual, textual, auditory and microexpression features dataset.
        Is quite slow as processing of image is applied on the fly.
    """

    def __init__(self, root_annotation, image_annotation,
                 root_dir_images, root_audio, root_text,
                 pretrained_audio=lambda x: x, transform=None):
        """
        Args:
            root_annotation (string): Path to the csv file with annotations on guestures
            image_annotation (string): Directory with annotation relating to videos (start and end index in imageloader).
            root_images: root directory for images
            root_audio: root directory containing data extracted from audio of videos
            root_text: transformed vectors from text
            transform (callable, optional): Optional transform to be applied
                on a sample images of video
        """
        self.annotation = pd.read_csv(root_annotation)
        self.image_annotation = pd.read_csv(image_annotation, index_col='name')
        self.images = torchvision.datasets.ImageFolder(root_dir_images, transform=preprocess)
        self.audio = pd.read_pickle(root_audio)
        self.text = pd.read_pickle(root_text)
        self.pretrained_audio = pretrained_audio
        self.transform = transform


    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):

        vid_name = self.annotation.iloc[idx, 0]
        start = self.image_annotation.loc[vid_name[:-4], 'start']
        end = self.image_annotation.loc[vid_name[:-4], 'end']
        images = video_loader(start, end, self.images)

        text = self.text.loc[vid_name, 'Embedding']
        #text = np.append(text, np.zeros((MAX_LEN_TEXT - text.shape[0], 300)), axis=0)

        audio = self.pretrained_audio(self.audio.loc[vid_name, 'Audio'])

        micro = self.annotation.iloc[idx, 1:-1].values.astype('float')

        self.sample = { 'audio': audio, 'microexpressions': micro, 'text': text, 'images': images}

        target = 0
        if self.audio.loc[vid_name, 'Label'] == 'Truthful':
            target = 1

        if self.transform:
            sample = self.transform(sample)

        return self.sample, target
