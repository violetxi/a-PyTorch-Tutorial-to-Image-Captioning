import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from datasets import *
from models import Encoder, DecoderWithAttention
from PIL import Image


# Parameters
checkpoint_path = 'bootstrapping_models/which-is-More/BEST_checkpoint_coco_1_cap_per_img_1_min_word_freq.pth.tar'
wordmap_path = 'data/bootstrapping_data/set_combined/WORDMAP_coco_1_cap_per_img_1_min_word_freq.json'
data_folder = 'data/caption_kids/set_combined/freq_1/'  # folder with data files saved by create_input_files.py
data_name = 'coco_1_cap_per_img_1_min_word_freq'  # base name shared by data files 
batch_size = 1
# Load word map
wordmap = json.load(open(wordmap_path))
print(wordmap)
# Load checkpoint
checkpoint = torch.load(checkpoint_path)
encoder = checkpoint['encoder']
encoder = encoder.cuda()
encoder.eval()
'''
decoder = checkpoint['decoder']
decoder = decoder.cuda()
decoder.eval()
'''
# Load training and evaluation set
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
trainset = TripletDataset(data_folder, data_name, 'TRAIN',
                          transform=transforms.Compose([normalize]))
trainset.load_triplets()
train_loader = torch.utils.data.DataLoader(trainset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=1,
                                           pin_memory=True)
# Create embeddings
embeddings_pos, embeddings_neg = [], []
for i, (imgs_neg, caps_neg, caplens_neg,
        imgs_anchor, caps_anchor, caplens_anchor,
        imgs_pos, caps_pos, caplens_pos) in enumerate(train_loader):
    # Negative embeddings
    imgs_neg = imgs_neg.cuda()
    caps_neg = caps_neg.cuda()
    caplens_neg = caplens_neg.cuda()
    embedding_neg = encoder(imgs_neg).contiguous().view(-1,)
    # Positive embeddings
    imgs_pos = imgs_pos.cuda()
    caps_pos = caps_pos.cuda()
    caplens_pos = caplens_pos.cuda()
    embedding_pos = encoder(imgs_pos).contiguous().view(-1,)
    embeddings_pos.append(embedding_pos.cpu().numpy())
    embeddings_neg.append(embedding_neg.cpu().numpy())

"Dimension reduction"
pca = PCA(n_components=100)
embeddings = embeddings_pos + embeddings_neg
pca.fit(embeddings)
embedding_neg_pca = pca.transform(embeddings_neg)
embedding_pos_pca = pca.transform(embeddings_pos)
# TSNE
tsne = TSNE(n_components=2)
embedding_tsne = tsne.fit_transform(embedding_neg_pca + embedding_pos_pca)
embedding_neg_tsne = embedding_tsne[ : len(embedding_tsne)//2]
embedding_pos_tsne = embedding_tsne[len(embedding_tsne)//2 : ]
np.save('pos_embeddings.npy', embedding_pos_tsne)
np.save('neg_embeddings.npy', embedding_neg_tsne)
plt.plot(embedding_neg_tsne, 'o', color='r', label='negative')
plt.plot(embedding_pos_tsne, 'x', color='b', label='positive')
#plt.legend()
plt.savefig('embedding_vis.png')
plt.show()
