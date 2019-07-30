import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def combine_map(file_path, added_maps):
    old_files = [f for f in os.listdir(file_path) if f.endswith('npy') and 'old' in f]
    new_files = [f for f in os.listdir(file_path) if f.endswith('npy') and 'new' in f]
    for f in old_files:
        digit = int(f[0])
        position = f[f.find('_')+1 : f.rfind('_')]
        added_maps[digit][position] += np.load(os.path.join(file_path, f))
    for f in new_files:
        digit = int(f[0])
        position = f[f.find('_')+1 : f.rfind('_')]
        added_maps[digit][position] += np.load(os.path.join(file_path, f))
    for digit in added_maps.keys():
        for pos in added_maps[digit].keys():
            plt.imshow(added_maps[digit][pos], cmap='RdGy')
            plt.colorbar()
            name = "{}_{}.png".format(digit, pos)
            plt.title(name)
            out_path = os.path.join(os.path.join(file_path, 'combined'), name)
            plt.savefig(out_path)
            plt.close()

##### Main #####
file_path = sys.argv[1]
added_maps = {}
# Regular
'''
for digit in [2, 3, 4]:
    if digit == 2:
        added_maps[digit] = {'ty_teens' : np.zeros((336, 336)), 'unit' : np.zeros((336, 336))}
    if digit == 3:
        added_maps[digit] = {'ty_teens' : np.zeros((336, 336)), 'unit' : np.zeros((336, 336)),
                             'hundred' : np.zeros((336, 336)), 'pre-hundred' : np.zeros((336, 336))}
    if digit == 4:
        added_maps[digit] = {'ty_teens' : np.zeros((336, 336)), 'unit' : np.zeros((336, 336)),
                             'hundred' : np.zeros((336, 336)), 'pre-hundred' : np.zeros((336, 336)),
                             'thousand' : np.zeros((336, 336)), 'pre-thousand' : np.zeros((336, 336))}
'''
# T_sep
for digit in [2, 3, 4]:
    if digit == 2:
        added_maps[digit] = {'pre_tys' : np.zeros((336, 336)), 'ty_teens' : np.zeros((336, 336)),
                             'unit' : np.zeros((336, 336))}
    if digit == 3:
        added_maps[digit] = {'pre_tys' : np.zeros((336, 336)), 'ty_teens' : np.zeros((336, 336)),
                             'unit' : np.zeros((336, 336)), 'hundred' : np.zeros((336, 336)),
                             'pre-hundred' : np.zeros((336, 336))}
    if digit == 4:
        added_maps[digit] = {'pre_tys' : np.zeros((336, 336)), 'ty_teens' : np.zeros((336, 336)),
                             'unit' : np.zeros((336, 336)), 'hundred' : np.zeros((336, 336)),
                             'pre-hundred' : np.zeros((336, 336)), 'thousand' : np.zeros((336, 336)),
                             'pre-thousand' : np.zeros((336, 336))}

# Combine maps
combine_map(file_path, added_maps)

