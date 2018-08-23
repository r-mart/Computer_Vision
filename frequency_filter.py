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


def low_pass_filter(img, low_pass_fraction = 0.3, **kwargs):

    rows, cols = img.shape
    crow, ccol = rows // 2 , cols // 2 # center point

    row_lp = int(crow * low_pass_fraction)
    col_lp = int(ccol * low_pass_fraction)    

    # Transform to Fourier space
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    # low pass with elliptical kernel
    mask = np.zeros((rows, cols),np.uint8)
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*col_lp, 2*row_lp)) # col, row are inverted between openCV and numpy
    mask[crow-row_lp:crow+row_lp, ccol-col_lp:ccol+col_lp] = ellipse

    fshift = fshift * mask

    # Transform back
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back, magnitude_spectrum  


def high_pass_filter(img, high_pass_fraction = 0.3, **kwargs):

    rows, cols = img.shape
    crow, ccol = rows // 2 , cols // 2 # center point

    row_hp = int(crow * high_pass_fraction)
    col_hp = int(ccol * high_pass_fraction)

    # Transform to Fourier space
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    # manipulation

    # high pass with elliptical kernel
    mask = np.ones((rows, cols), np.uint8)
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*col_hp, 2*row_hp)) # col, row are inverted between openCV and numpy
    ellipse = np.logical_not(ellipse)
    mask[crow-row_hp:crow+row_hp, ccol-col_hp:ccol+col_hp] = ellipse

    fshift = fshift * mask

    # Transform back
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back, magnitude_spectrum     


def middle_pass_filter(img, high_pass_fraction = 0.3, low_pass_fraction = 0.3, **kwargs):

    rows, cols = img.shape
    crow, ccol = rows // 2 , cols // 2 # center point

    row_lp = int(crow * low_pass_fraction)
    col_lp = int(ccol * low_pass_fraction)        

    row_hp = int(crow * high_pass_fraction)
    col_hp = int(ccol * high_pass_fraction)

    # Transform to Fourier space
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    # manipulation

    # low pass with elliptical kernel
    mask_lp = np.zeros((rows, cols),np.uint8)
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*col_lp, 2*row_lp)) # col, row are inverted between openCV and numpy
    mask_lp[crow-row_lp:crow+row_lp, ccol-col_lp:ccol+col_lp] = ellipse    

    # high pass with elliptical kernel
    mask_hp = np.ones((rows, cols), np.uint8)
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*col_hp, 2*row_hp)) # col, row are inverted between openCV and numpy
    ellipse = np.logical_not(ellipse)
    mask_hp[crow-row_hp:crow+row_hp, ccol-col_hp:ccol+col_hp] = ellipse

    mask = np.logical_and(mask_lp, mask_hp)

    fshift = fshift * mask

    # Transform back
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # prepare mask for inspection
    mask = mask.astype(np.uint8)
    mask[np.where(mask == 1)] = 255

    return img_back, magnitude_spectrum, mask      


def main():

    # Parameter #
    params = dict(
        high_pass_fraction = 0.15, # only frequencies higher than this fraction of the fourier domain image will pass
        low_pass_fraction = 0.4 # only frequencies lower than this fraction of the fourier domain image will pass
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

            low_pass, fourier_spectrum = low_pass_filter(img, **params)         
            high_pass, _ = high_pass_filter(img, **params) 
            middle_pass, _, middle_mask = middle_pass_filter(img, **params)

            # OTSU Thresholding
            low_pass = low_pass.astype(np.uint8)
            ret, low_pass_otsu = cv2.threshold(low_pass, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            middle_pass = middle_pass.astype(np.uint8)
            ret, middle_pass_otsu = cv2.threshold(middle_pass, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            cv2.imwrite(os.path.join(output_path, input_name), img_original)
            cv2.imwrite(os.path.join(output_path, input_name_base + "_fourier_spectrum.tiff"), fourier_spectrum)
            cv2.imwrite(os.path.join(output_path, input_name_base + "_low_pass.tiff"), low_pass)
            cv2.imwrite(os.path.join(output_path, input_name_base + "_high_pass.tiff"), high_pass)
            cv2.imwrite(os.path.join(output_path, input_name_base + "_middle_pass.tiff"), middle_pass)
            cv2.imwrite(os.path.join(output_path, input_name_base + "_middle_mask.tiff"), middle_mask)
            cv2.imwrite(os.path.join(output_path, input_name_base + "_middle_pass_otsu.tiff"), middle_pass_otsu)
            cv2.imwrite(os.path.join(output_path, input_name_base + "_low_pass_otsu.tiff"), low_pass_otsu)

            duration = time.time() - start
            logging.info("Processed {0} (Duration: {1:.3f} s)".format(input_name, duration)) 
            counter += 1                       

    logging.info("Processed {} images in total".format(counter))
    print("Processing done. See log file in '{}' for more details".format(output_path))

if __name__ == "__main__":
    main()