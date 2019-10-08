"""
This script is from:
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/datasets.py
"""
import torch
from torch.utils.data import Dataset
import h5py
import json
import os


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """
    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size


# Used for Which-Is-More training set. Each Triplet has an achor, which
# is less than the positive example and greater than the negative example
class TripletDataset(Dataset):
    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform
        # Total number of datapoints
        self.dataset_size = len(self.captions) - 2

    def __getitem__(self, i):
        # Get triplet images
        img_neg = torch.FloatTensor(self.negatives[i // self.cpi] / 255.)
        img_anchor = torch.FloatTensor(self.anchors[i // self.cpi] / 255.)
        img_pos = torch.FloatTensor(self.positives[i // self.cpi] / 255.)
        
        if self.transform is not None:
            img_neg = self.transform(img_neg)
            img_anchor = self.transform(img_anchor)
            img_pos = self.transform(img_pos)
            
        # Triplet captions
        caption_neg = torch.LongTensor(self.neg_captions[i])
        caption_anchor = torch.LongTensor(self.anchor_captions[i])
        caption_pos = torch.LongTensor(self.pos_captions[i])
        # Ttriplet caplengths
        caplen_neg = torch.LongTensor([self.neg_caplens[i]])
        caplen_anchor = torch.LongTensor([self.anchor_caplens[i]])
        caplen_pos = torch.LongTensor([self.pos_caplens[i]])

        return img_neg, caption_neg, caplen_neg, img_anchor, caption_anchor, caplen_anchor, img_pos, caption_pos, caplen_pos

    # Minus 2 since we are only returning triplets
    def __len__(self):
        return self.dataset_size


    # Load dataset into the triplets (Anchor, Negative & Positive)
    def load_triplets(self):
        # Images
        self.anchors, self.positives, self.negatives = [], [], []
        # Captions
        self.anchor_captions = []
        self.pos_captions = []
        self.neg_captions = []
        # Caption lengths
        self.anchor_caplens = []
        self.pos_caplens = []
        self.neg_caplens = []

        for i in range(self.dataset_size):
            # Image triplets
            self.negatives.append(self.imgs[i / self.cpi])
            self.anchors.append(self.imgs[(i+1) / self.cpi])
            self.positives.append(self.imgs[(i+2) / self.cpi])
            # Triplet's captions
            self.neg_captions.append(self.captions[i])
            self.anchor_captions.append(self.captions[i+1])
            self.pos_captions.append(self.captions[i+2])
            # Triplets caplens
            self.neg_caplens.append(self.caplens[i])
            self.anchor_caplens.append(self.caplens[i+1])
            self.pos_caplens.append(self.caplens[i+2])
        
