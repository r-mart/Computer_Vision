""" Lattice extraction using a rank filter

Assumptions:
 - Input images are binary
 - Images are black on white (meaning foreground elements are black)
"""

import os
import datetime
import time
import cv2
import numpy as np
import json
import logging
import argparse
from scipy.ndimage.filters import rank_filter

import utils


def extract_lattice(img, filter_length = 30, rank = 1, size_close=5, **kwargs):
    """ Actual lattice extraction """   

    maxed_rows = rank_filter(img, -rank, size=(1, filter_length))
    maxed_cols = rank_filter(img, -rank, size=(filter_length, 1))
    filtered = np.maximum(np.maximum(img, maxed_rows), maxed_cols)
    lattice = np.minimum(maxed_rows, maxed_cols) 

    return lattice   


def main():

    # Parameter #
    params = dict(
        size_open = 2,
        size_close = 8,    
        filter_length = 40,
        rank = 1
    )
    #############

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="./Input/demo", help="Path to the folder containing the input images.")
    parser.add_argument("--output_path", default="./Output/demo", help="Path to the folder which will contain the output.")
    parser.add_argument("--param_file", default="", help="Name of a parameter file in the input folder. Will be used to override the local param dictionary.")    
    args = parser.parse_args()

    # Preparation
    time_stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    current_file = os.path.splitext(os.path.basename(__file__))[0]

    input_path = args.input_path
    output_path = os.path.join(args.output_path, current_file + "_" + time_stamp)

    if not os.path.exists(output_path):
        os.makedirs(output_path) 

    # set up logging
    logging.basicConfig(filename=os.path.join(output_path, current_file + '.log'), level=logging.DEBUG, 
                        format='%(asctime)s - %(levelname)s: %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')     

    logging.info("Current git revision: {}".format(utils.get_git_revision_hash()))

    # override parameter if external ones are given
    param_path = os.path.join(input_path, args.param_file)
    if os.path.isfile(param_path):
        with open(param_path, "r") as param_file:
            params = json.load(param_file)
        logging.info("Using parameter given in {}".format(param_path))
    else:
        logging.info("Using local parameter")          

    # dump used parameter
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        json.dump(params, f, sort_keys=True, indent=4, )    
        f.write('\n')          

    print("Start processing...")
    counter = 0
    # Loop through all images in input path
    for root, dirs, files in os.walk(input_path):
        for input_name in files: 

            start = time.time()
            input_name_base = os.path.splitext(os.path.basename(input_name))[0]

            img_original = cv2.imread(os.path.join(input_path, input_name))
            if img_original is None: # reading failed (e.g. file is not an image)
                continue

            img = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)  
            img = utils.prepare_for_morph_filter(img)
            denoised = utils.morph_denoise(img, **params)          
            denoised = utils.restore_after_morph_filter(denoised)

            lattice = extract_lattice(denoised, **params)         

            cv2.imwrite(os.path.join(output_path, input_name), img_original)
            cv2.imwrite(os.path.join(output_path, input_name_base + "_denoised.tiff"), denoised)
            cv2.imwrite(os.path.join(output_path, input_name_base + "_lattice.tiff"), lattice)

            duration = time.time() - start
            logging.info("Processed {0} (Duration: {1:.3f} s)".format(input_name, duration)) 
            counter += 1                       

    logging.info("Processed {} images in total".format(counter))
    print("Processing done. See log file in '{}' for more details".format(output_path))

if __name__ == "__main__":
    main()