"""                                                                                                                                                           
This script is from: 
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/create_input_files.py

"""

from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path='/l/vision/v5/ziyxiang/number_modeling/NumberModeling/bootstrap_dataset/caption_annotation/kid_combined.json',
                       image_folder='/l/vision/v5/ziyxiang/number_modeling/NumberModeling/',
                       captions_per_image=1,
                       min_word_freq=1,
                       output_folder='data/bootstrapping_data/set_combined/',
                       max_len=50)
