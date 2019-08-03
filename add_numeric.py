""" 
Add the following information in inference result files:
1). target_true_number, target_predicted_number
2). foil_true_number, foil_predicted_number
"""

import os
import argparse
import pandas as pd



# Convert words of numnbers to numerics 
# (Source code: https://stackoverflow.com/questions/493174/is-there-a-way-to-convert-number-words-to-integers)
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
            #raise Exception("Illegal word: " + word)
            return "N/A"
        scale, increment = numwords[word]
        current = current * scale + increment
        if scale > 100:
            result += current
            current = 0
    return result + current

# Add numeric value into the result files
def add_numeric(res_dir):
    res_files = os.listdir(res_dir)
    for f in res_files:
        f_path = os.path.join(res_dir, f)
        results = pd.DataFrame.from_csv(f_path)
        true_target_nums, true_foil_nums, pred_target_nums, pred_foil_nums = [], [], [], []
        for index, row in results.iterrows():
            true_target_nums.append(word2num(convert_number_label(row["desired_label"])))
            true_foil_nums.append(word2num(convert_number_label(row["foil_word"])))
            pred_target_nums.append(word2num(convert_number_label(row["target_model_generated_label"])))
            pred_foil_nums.append(word2num(convert_number_label(row["foil_model_generated_label"])))
        added_results = results.assign(true_target_numbers=true_target_nums, true_foil_numbers=true_foil_nums, 
                                  predicted_target_numbers=pred_target_nums, predicted_foil_numbers=pred_foil_nums)
        added_results.to_csv(f_path)

# Convert number
import math
def convert_number_label(label):
    # If no number is recognized, it is denoted as 0
    if label is not isinstance(label, str):
        return "zero"
    elif ' ty' in label:
        return label.replace(' ty', 'ty')
    elif ' teen' in label:
        return label.replace(' teen', 'teen')
    else:
        return label


parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')
parser.add_argument('--res_dir', '-r', help='Directory where all the results were saved.')
args = parser.parse_args()
add_numeric(args.res_dir)
