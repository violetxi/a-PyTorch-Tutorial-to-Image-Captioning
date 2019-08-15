import os
import json
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
def get_raw_att_file(image, prediction, label, model):
    att_file_dir = os.path.join(args.raw_attention, 
                                os.path.join(model, args.type))
    # attention regions output
    att_regions = []
    # Conver string to list of words
    if type(prediction) == str:
        prediction = prediction.split(" ")
        for i in range(len(prediction)):
            # Load attention map
            word = prediction[i]
            if word != "<end>" and word != "<start>":
                att_file = "{}_{}_{}.npy".format(image, word, i+1)
                att_map = np.load(os.path.join(att_file_dir, att_file))
                # Get num of digits
                num_digit = get_digit_num(label)
                # Attention region
                att_regions.append(get_max_region(att_map, num_digit))

    return att_regions

        
# Given raw attention map and # of digits return which region it belongs to
def get_max_region(att_map, num_digit):
    r, c = att_map.shape
    if num_digit == 1:
        return att_region_map["unit"]
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
    return att_region_map[regions[np.argmax(atts)]]


# Generate attention region result for each file, return a pd.DF object including:
# Label, Prediction, Word_i_attention_region (where i is nth word in label)
# Since we don't care whether an image is foil or target convert 1 row in res to 2 rows
# in att_region
def get_attention_region_single(res, model):
    # Read prediction results and initialize attention_region result
    res_df = pd.read_csv(res)
    rows = []
    # Get a list of attention map file names
    for index, row in res_df.iterrows():
        # Label, prediction and im_name for target
        label_1 = row["desired_label"]
        prediction_1 = row["target_model_generated_label"]
        image_1 = row["target_image"]
        att_regions_1 = get_raw_att_file(image_1, 
                                         prediction_1, 
                                         label_1, model)        
        row_1 = [image_1, label_1, prediction_1] + [att_region for att_region in att_regions_1]
        row_1 += ["N/A" for i in range(9-len(row_1))]
        rows.append(row_1)
        # Label, prediction and im_name for foil
        label_2 = row["foil_word"]
        prediction_2 = row["foil_model_generated_label"]
        image_2 = row["foil_image"]
        att_regions_2 = get_raw_att_file(image_2, 
                                         prediction_2, 
                                         label_2, model)
        row_2 = [image_2, label_2, prediction_2] + [att_region for att_region in att_regions_2]
        row_2 += ["N/A" for i in range(9-len(row_2))]
        rows.append(row_2)
    att_region_df = pd.DataFrame(rows,
                                 columns=["Image", "Label", "Prediction"] + 
                                 ["Word_{}_attention_region".format(i) for i in range(1, 7)])
    return att_region_df


# For each model, generate a csv file containing results for attention region
def get_attention_region(res_dir):
    for i in range(1, args.num_of_models+1):        
        model = "model_{}".format(i)
        res = os.path.join(res_dir,                               
                           "res_{}_{}.csv".format(args.type, model))
        att_region_df = get_attention_region_single(res, model)
        input_type = args.raw_attention[args.raw_attention.find('/') + 1 :]
        out_dir = os.path.join(args.out, input_type)
        try:
            os.mkdir(out_dir)
        except:
            pass
        out = os.path.join(out_dir,
                           "attention_region_{}_{}.csv".format(args.type, model))
        print("Saving to {}".format(out))
        #att_region_df.to_csv(out)




parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption.')
parser.add_argument('--res_dir', '-r', help='Directory where all the results were saved.')    
parser.add_argument('--attention_region_map', '-a', help='Attention region map file.')
parser.add_argument('--raw_attention', '-ra', help="Path to raw attention map directories.")
parser.add_argument('--num_of_models', '-n', type=int, help="Number of models.")
parser.add_argument('--type', '-t', help="Whether it's new or old.")
parser.add_argument('--out', '-o', help="Output directory root.")
args = parser.parse_args()
# Load attention region map
att_region_map = json.load(open(args.attention_region_map))
get_attention_region(args.res_dir)
