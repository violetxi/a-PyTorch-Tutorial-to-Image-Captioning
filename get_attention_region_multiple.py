import os
import json
import glob
import argparse
import numpy as np
import pandas as pd



# Decide how many digits there are in a number given the label
def get_digit_num(label):
    if "thousand" in label:
        return 4
    elif "hundred" in label:
        return 3
    elif "teen" in label or "ty" in label:
        return 2
    else:
        return 1

# Return attention regions for each file name given image name, prediciton, 
# label and model name for each word
def get_raw_att_file(prediction, label, im_att_files):
    word_region_att = {}
    #if type(prediction) == str:
    for i in range(len(prediction)):
        # Load attention map
        word = prediction[i]
        if word != "<end>" and word != "<start>":
            att_file = [f for f in im_att_files if word in f][0]
            att_map = np.load(att_file)
            # Get num of digits            
            num_digit = get_digit_num(label)
            # Attention region
            word_region_att[i] = get_region_scores(att_map, num_digit)
    return word_region_att

        
# Given raw attention map and # of digits return which region it belongs to
def get_region_scores(att_map, num_digit):
    # attention regions output
    att_region_scores = {}
    r, c = att_map.shape
    if num_digit == 1:
        regions = ["unit"]
        atts = [np.sum(att_map)]
    elif num_digit == 2:
        ty_teen_region_att = np.sum(att_map[:, :int(c / num_digit)])        
        unit_region_att = np.sum(att_map[:, int(c / num_digit):])
        regions = ["teen_ty", "unit"]
        atts = [ty_teen_region_att, unit_region_att]
    elif num_digit == 3:
        pre_hundred_region_att = np.sum(att_map[:, :int(c / num_digit)])
        ty_teen_region_att = np.sum(att_map[:, int(c / num_digit) : int(c*2 / num_digit)])
        unit_region_att = np.sum(att_map[:, int(c*2 / num_digit) : ])
        regions = ["pre-hundred", "teen_ty", "unit"]        
        atts = [pre_hundred_region_att, ty_teen_region_att, unit_region_att]
    else:
        pre_thousand_region_att = np.sum(att_map[:, :int(c / num_digit)])
        pre_hundred_region_att = np.sum(att_map[:, int(c / num_digit) : int(c*2 / num_digit)])
        ty_teen_region_att = np.sum(att_map[:, int(c*2 / num_digit) : int(c*3 / num_digit)])
        unit_region_att = np.sum(att_map[:, int(c*3 / num_digit) : ])
        regions = ["pre-thousand", "pre-hundred", "teen_ty", "unit"]
        atts = [pre_thousand_region_att, pre_hundred_region_att, 
                ty_teen_region_att, unit_region_att]
    for i in range(len(regions)):
        region, att = regions[i], atts[i]
        att_region_scores[att_region_map[region]] = att / np.sum(att_map)
    return att_region_scores


# Generate attention region result for each file, return a pd.DF object including:
# Label, Prediction, Word_i_attention_region (where i is nth word in label)
# Since we don't care whether an image is foil or target convert 1 row in res to 2 rows
# in att_region
def get_attention_region_single(res):
    # Get all attention files for the model
    att_files = glob.glob("{}/*npy".format(res))
    rows = []
    for im in sorted(gt.keys()):
        # retrieve attention files related to current images
        im_att_files = [f for f in att_files if f[f.rfind('/')+1:].startswith(im)]
        label = gt[im]
        prediction = [f[f.rfind('/')+1:][f[f.rfind('/')+1:].find('_')+1: f[f.rfind('/')+1:].rfind('_')] for f in im_att_files]
        # In the format of {word_i : {regions_j : score, ...}}
        att_region_scores = get_raw_att_file(prediction, label, im_att_files)
        for word in att_region_scores.keys():
            row = [im, label, " ".join(prediction), word]
            scores = [0, 0, 0, 0]
            for att_region in att_region_scores[word].keys():                
                scores[att_region-1] = att_region_scores[word][att_region]
            rows.append(row + scores)

    att_region_df = pd.DataFrame(rows,
                                 columns=["Image", "Label", "Prediction", "Word",
                                          "Thousand_score", "Hundred_score", 
                                          "Ty/Teen_score", "Unit_score"])
    return att_region_df

# For each model, generate a csv file containing results for attention region
def get_attention_region(raw_att_dir):
    for i in range(1, args.num_of_models+1):
        model = "model_{}".format(i)
        " For new/old set"
        #raw_att_path = os.path.join(raw_att_dir, "{}/{}".format(model, args.type))
        #res = os.path.join(args.out, "att_{}_{}.csv".format(args.type, model))
        "For new test set"
        raw_att_path = os.path.join(raw_att_dir, model)
        res = os.path.join(args.out, "res_{}.csv".format(model))
        # Get proabilities for attention regions
        att_region_df = get_attention_region_single(raw_att_path)
        out = os.path.join(args.out,
                           "attention_region_{}_{}.csv".format(args.type, model))
        print("Saving to {}".format(out))
        att_region_df.to_csv(out)



parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption.')
parser.add_argument('--gt', '-g', help='Ground truch json file.')    
parser.add_argument('--attention_region_map', '-a', help='Attention region map file.')
parser.add_argument('--raw_attention', '-ra', help="Path to raw attention map directories.")
parser.add_argument('--num_of_models', '-n', type=int, help="Number of models.")
parser.add_argument('--type', '-t', help="Whether it's new or old.")
parser.add_argument('--out', '-o', help="Output directory where csv files will be saved.")
args = parser.parse_args()
# Load attention region map
att_region_map = json.load(open(args.attention_region_map))
gt= json.load(open(args.gt))
get_attention_region(args.raw_attention)
