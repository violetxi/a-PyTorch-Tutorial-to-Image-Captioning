import sys
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import Encoder, DecoderWithAttention
from datasets import *
from utils import *
from losses import *
from nltk.translate.bleu_score import corpus_bleu

# Data parameters
data_folder = 'data/caption_kids/set_combined/freq_1/'  # folder with data files saved by create_input_files.py
data_name = 'coco_1_cap_per_img_1_min_word_freq'  # base name shared by data files

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
# Loss parameters
margin = 1e5

# Training parameters
start_epoch = 0
epochs = 1000  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 1
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-6  # learning rate for encoder if fine-tuning
decoder_lr = 1e-6  # learning rate for decoder
grad_clip = 10.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = True  # fine-tune encoder?
checkpoint = 'bootstrapping_models/which-is-N/model_1/BEST_checkpoint_coco_1_cap_per_img_1_min_word_freq.pth.tar'#None  # path to checkpoint, None if none
lowest_loss = float('Inf')




def main():
    """
    Training and validation.
    """

    global lowest_loss, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=len(word_map),
                                       dropout=dropout)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)
        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        #start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if fine_tune_encoder is True and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # Move to GPU, if available
    decoder = decoder.cuda()
    encoder = encoder.cuda()
    # Loss function
    criterion = TripletLoss(margin).cuda()

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # Training set
    trainset = TripletDataset(data_folder, data_name, 'TRAIN', 
                              transform=transforms.Compose([normalize]))
    trainset.load_triplets()
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=batch_size, 
                                               shuffle=True, 
                                               num_workers=workers, 
                                               pin_memory=True)
    # Validation
    valset = TripletDataset(data_folder, data_name, 'VAL', 
                            transform=transforms.Compose([normalize]))
    valset.load_triplets()
    val_loader = torch.utils.data.DataLoader(valset,
                                             batch_size=batch_size, 
                                             shuffle=True, 
                                             num_workers=workers, 
                                             pin_memory=True)
    
    # Epochs
    for epoch in range(start_epoch, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 500:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 100 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)
        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)
        # One epoch's validation
        recent_val_loss = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion)
        # Check if there was an improvement
        is_best = recent_val_loss < lowest_loss
        lowest_loss = min(recent_val_loss, lowest_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, 
                        encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_val_loss, is_best)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy
    start = time.time()

    # Batches
    for i, (imgs_neg, caps_neg, caplens_neg, 
            imgs_anchor, caps_anchor, caplens_anchor, 
            imgs_pos, caps_pos, caplens_pos) in enumerate(train_loader):
        data_time.update(time.time() - start)
        # Negative embeddings
        imgs_neg = imgs_neg.cuda()
        caps_neg = caps_neg.cuda()
        caplens_neg = caplens_neg.cuda()
        imgs_neg = encoder(imgs_neg)
        cell_states_neg, scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs_neg, caps_neg, caplens_neg)
        # Anchor embeddings
        imgs_anchor = imgs_anchor.cuda()
        caps_anchor = caps_anchor.cuda()
        caplens_anchor = caplens_anchor.cuda()
        imgs_anchor = encoder(imgs_anchor)
        cell_states_anchor, scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs_anchor, caps_anchor, caplens_anchor)
        # Positive embeddings
        imgs_pos = imgs_pos.cuda()
        caps_pos = caps_pos.cuda()
        caplens_pos = caplens_pos.cuda()
        imgs_pos = encoder(imgs_pos)
        cell_states_pos, scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs_pos, caps_pos, caplens_pos)
        # Calculate loss
        loss = criterion(imgs_neg.contiguous().view(batch_size, -1), 
                         imgs_anchor.contiguous().view(batch_size, -1), 
                         imgs_pos.contiguous().view(batch_size, -1))
        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        losses.update(loss.item(), sum(decode_lengths))
        batch_time.update(time.time() - start)
        start = time.time()
        
        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))

def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.                                                                                                                                                                        
    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: loss on validation set
    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()
        
    # Keep track of metrics
    batch_time = AverageMeter()
    losses = AverageMeter()    
    start = time.time()
    # Batches
    for i, (imgs_neg, caps_neg, caplens_neg, 
            imgs_anchor, caps_anchor, caplens_anchor, 
            imgs_pos, caps_pos, caplens_pos) in enumerate(val_loader):
        # Negative embeddings
        imgs_neg = imgs_neg.cuda()
        caps_neg = caps_neg.cuda()
        caplens_neg = caplens_neg.cuda()
        imgs_neg = encoder(imgs_neg)
        cell_states_neg, scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs_neg, caps_neg, caplens_neg)
        # Anchor embeddings
        imgs_anchor = imgs_anchor.cuda()
        caps_anchor = caps_anchor.cuda()
        caplens_anchor = caplens_anchor.cuda()
        imgs_anchor = encoder(imgs_anchor)
        cell_states_anchor, scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs_anchor, caps_anchor, caplens_anchor)
        # Positive embeddings
        imgs_pos = imgs_pos.cuda()
        caps_pos = caps_pos.cuda()
        caplens_pos = caplens_pos.cuda()
        imgs_pos = encoder(imgs_pos)
        cell_states_pos, scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs_pos, caps_pos, caplens_pos)
        # Calculate loss
        loss = criterion(imgs_neg.contiguous().view(batch_size, -1), 
                         imgs_anchor.contiguous().view(batch_size, -1), 
                         imgs_pos.contiguous().view(batch_size, -1))
        # Add doubly stochastic attention regularization
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        # Keep track of metrics
        losses.update(loss.item(), sum(decode_lengths))
        batch_time.update(time.time() - start)
        start = time.time()
        # Print status
        if i % print_freq == 0:
            print('Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(loss=losses))
        return losses.avg
        

if __name__ == '__main__':
    main()
