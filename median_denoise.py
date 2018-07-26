import os
import datetime
import time
import cv2
import numpy as np
import json
import logging
import argparse

import utils


def median_denoise(img, median_kernel_size, **params):

    img = cv2.medianBlur(img, median_kernel_size)

    return img

def main():

    # Parameter #
    params = dict(    
        median_kernel_size = 3   
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
            img = median_denoise(img, **params)            

            cv2.imwrite(os.path.join(output_path, input_name), img_original)
            cv2.imwrite(os.path.join(output_path, input_name_base + "_processed.tiff"), img)

            duration = time.time() - start
            logging.info("Processed {0} (Duration: {1:.3f} s)".format(input_name, duration)) 
            counter += 1                       

    logging.info("Processed {} images in total".format(counter))
    print("Processing done. See log file in '{}' for more details".format(output_path))

if __name__ == "__main__":
    main()