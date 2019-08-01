""" 
Average attention map results for each position over all 4-digit, 3-digit 
and 2-digit numbers for each and all the models. Results are separated 
into old and new two folders.
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load ground truth information and save number of digit 
# information for each image
def load_gt():
    digit_images = {}
    for img in gt_data.keys():
        if 'thousand' in gt_data[img]:
            digit_images[img] = 4
        elif 'hundred' in gt_data[img]:
            digit_images[img] = 3
        elif 'ty' in gt_data[img] or 'teen' in gt_data[img] or 'eleven' in gt_data[img] or 'twelve' in gt_data[img] or 'ten' in gt_data[img]:
            digit_images[img] = 2
        else:
            digit_images[img] = 1
    return digit_images


# Compute avg attention map at each attention position
def compute_attention_map(digit_imgs, att_map_dir, out):
    model_att_maps_sum = {}
    # Only use correctly recognized numbers
    #model_images_valid = unqualified_list(args.res_dir)
    for model in os.listdir(att_map_dir):
        # Keep track of current map
        added_maps = {}
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
        # Go trhough the files
        model_path = os.path.join(att_map_dir, os.path.join(model, args.set_type))
        att_map_files = sorted(os.listdir(model_path), key=lambda x: int(x[:x.find('.')]))
        att_map_dict = {att_map_file : np.load(os.path.join(model_path, att_map_file))
                        for att_map_file in att_map_files}        
        # Break map down to each digit
        digit_att_map_files = {digit : [] for digit in [2, 3, 4]}
        for img in gt_data.keys():
            # Check if images were correctly recognized
            #if img in model_images_valid[model]:
            # Use all images (for Untrained cases)
            try:
                # Retrieve all the map related to an image
                img_att_map_files = [att_map_file for att_map_file in att_map_files if att_map_file[ : att_map_file.find('_')] == img]            
                digit = digit_imgs[img]            
                if digit == 2:
                    i = 0
                    while i < len(img_att_map_files):
                        cur_map_file = img_att_map_files[i]
                        cur_map = np.load(os.path.join(model_path, cur_map_file))
                        if 'ty' in cur_map_file or 'teen' in cur_map_file or 'ten' in cur_map_file or 'eleven' in cur_map_file or 'twelve' in cur_map_file:
                            added_maps[2]['ty_teens'] += cur_map
                        elif i == len(img_att_map_files) - 1 and 'and' not in cur_map_file:
                            added_maps[2]['unit'] += cur_map
                        i += 1
                elif digit == 3:
                    i = 0
                    while i < len(img_att_map_files):
                        cur_map_file = img_att_map_files[i]
                        cur_map = np.load(os.path.join(model_path, cur_map_file))
                        if 'hundred' in cur_map_file:
                            added_maps[3]['hundred'] += cur_map
                            added_maps[3]['pre-hundred'] += np.load(os.path.join(model_path, img_att_map_files[i-1]))
                        elif 'ty' in cur_map_file or 'teen' in cur_map_file or 'ten' in cur_map_file or 'eleven' in cur_map_file or 'twelve' in cur_map_file:
                            added_maps[3]['ty_teens'] += cur_map
                        elif i == len(img_att_map_files) - 1 and 'and' not in cur_map_file:
                            added_maps[3]['unit'] += cur_map
                        i += 1
                elif digit == 4:
                    i = 0
                    while i < len(img_att_map_files):
                        cur_map_file = img_att_map_files[i]
                        cur_map = np.load(os.path.join(model_path, cur_map_file))
                        if 'thousand' in cur_map_file:
                            added_maps[4]['thousand'] += cur_map
                            added_maps[4]['pre-thousand'] += np.load(os.path.join(model_path, img_att_map_files[i-1]))
                        if 'hundred' in cur_map_file:
                            added_maps[4]['hundred'] += cur_map
                            added_maps[4]['pre-hundred'] += np.load(os.path.join(model_path, img_att_map_files[i-1]))
                        elif 'ty' in cur_map_file or 'teen' in cur_map_file or 'ten' in cur_map_file or 'eleven' in cur_map_file or 'twelve' in cur_map_file:
                            added_maps[4]['ty_teens'] += cur_map
                        elif i == len(img_att_map_files) - 1 and 'and' not in cur_map_file:
                            added_maps[4]['unit'] += cur_map
                        i += 1                        
                model_att_maps_sum[model] = added_maps
            except:
                pass

    ''' Finished with reading raw maps '''
    # Add all the heat maps together for each model (final results need to be / num of images per digit * number of models
    digit_att_map = {2 : {'ty_teens' : [], 'unit' : []}, 
                     3 : {'pre-hundred' : [], 'hundred' : [], 'ty_teens' : [], 'unit' : []}, 
                     4 : { pos : [] for pos in attention_positions}}
    for digit in [2, 3, 4]:
        positions = list(digit_att_map[digit].keys())
        # Keep track of attention map
        for model in model_att_maps_sum:        
            for pos in positions:            
                digit_att_map[digit][pos].append(model_att_maps_sum[model][digit][pos])
    # Add all the maps for each position
    digit_att_map_sum = {digit : {pos : np.zeros((336, 336)) for pos in digit_att_map[digit].keys()} for digit in digit_att_map.keys()}    
    for digit in digit_att_map.keys():
        for pos in digit_att_map[digit].keys():
            for att_map in digit_att_map[digit][pos]:
                digit_att_map_sum[digit][pos] += att_map

    # Count number of model and number of images per digit and normalize the map
    num_models = len(os.listdir(att_map_dir))
    all_digits = list(digit_imgs.values())
    digit_im_count = {digit : all_digits.count(digit) for digit in digit_att_map_sum.keys()}
    for digit in digit_att_map_sum.keys():
        for pos in digit_att_map_sum[digit]:
            digit_att_map_sum[digit][pos] /= (num_models * digit_im_count[digit])            
            plt.imshow(digit_att_map_sum[digit][pos], cmap='RdGy')
            plt.colorbar()
            name = "{}_{}.png".format(digit, pos)
            out = os.path.join(args.out, "{}_{}_{}.npy".format(digit, pos, args.set_type))
            np.save(out, digit_att_map_sum[digit][pos])
            plt.title(name)
            plt.savefig(os.path.join(args.out, name))
            plt.close()
            #plt.show()

# Decide which numbers should be used remove numbers images that 
# are incorrectly recognized and numbers less than 20 (two digit)
# For each model which numbers are used to produce map
def unqualified_list(result_dir):
    res_files = [f for f in os.listdir(result_dir) if f.endswith('csv') and args.set_type in f]
    model_valid_number = {}
    for res_file in res_files:
        model = res_file[res_file.find('model') : res_file.find('.')]
        if model not in model_valid_number.keys():
            model_valid_number[model] = []
        # Open and read the result file
        res = pd.DataFrame.from_csv(os.path.join(result_dir, res_file))
        for index, row in res.iterrows():
            target_image, foil_image = row["target_image"], row["foil_image"]
            target_num_true, target_num_pred = row["true_target_numbers"], row["predicted_target_numbers"]
            foil_num_true, foil_num_pred = row["true_foil_numbers"], row["predicted_foil_numbers"]
            # Include only correct predictions and exluding two digit number that are less than 20
            if target_num_true == target_num_pred:
                if not 10 <= target_num_true < 20:
                    model_valid_number[model].append(target_image)
            if foil_num_true == foil_num_pred:
                if not 10 <= foil_num_true < 20:
                    model_valid_number[model].append(foil_image)
    # Keep a list of valid number for each digit and save them
    f_out = open(os.path.join(args.out, "heat_map_img.txt"), 'w+')
    digit_nums = {2 : [], 3 : [], 4 : []}
    for model in model_valid_number.keys():
        for img in model_valid_number[model]:
            number = word2num(str(gt_data[img]))
            # Don't want single digit number
            if number > 9:
                digit_nums[len(str(number))].append(number)
    f_out.write(str(digit_nums))
    return model_valid_number


# Convert word to num
def word2num(textnum):
    numwords = {}
    # All the tokens
    units = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    scales = ["hundred", "thousand", "million", "billion", "trillion"]
    numwords["and"] = (1, 0)

    for idx, word in enumerate(units):    numwords[word] = (1, idx)
    for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
    for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)
    current = result = 0
    for word in textnum.split():
        if word not in numwords:
            return "N/A"
        scale, increment = numwords[word]
        current = current * scale + increment
        if scale > 100:
            result += current
            current = 0
    return result + current



# Command line arguments
parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')
parser.add_argument('--gt', '-g', help="Path to where ground truth is.")
parser.add_argument('--att_map_dir', '-ad', help="Path to where raw attention map is stored for each model.")
parser.add_argument('--set_type', '-s', help="If a set is new or old")
parser.add_argument('--out', '-o', help="Output directory to save averaged attention map.")
parser.add_argument('--res_dir', '-r', help="Directory where results are saved.")
args = parser.parse_args()
# Global variables
attention_positions = ['pre-thousand', 'thousand', 'pre-hundred', 'hundred', 'ty_teens', 'unit']
gt_data = json.load(open(args.gt))
# Main
digit_imgs = load_gt()
compute_attention_map(digit_imgs, args.att_map_dir, args.out)
