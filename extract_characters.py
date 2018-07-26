import os
import datetime
import time
import cv2
import numpy as np
import json
import logging
import argparse

import utils


def extract_chars(img_original):
    """ Extracts characters (blob-like objects) from a black on white image 
    
    Returns three images:
    - original image with hulls around the detected objects
    - original image with boxes around the detected objects
    - image containing only the content of the original image within the hulls (text-only)
    """

    img = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    img_hulls = img_original.copy() 
    img_boxes = img_original.copy() 
    img_text = img_original.copy() 

    mser = cv2.MSER_create()
    regions, bboxes = mser.detectRegions(img)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(img_hulls, hulls, 1, (0, 0, 255), 2)

    for bbox in bboxes:
        x, y, w, h = bbox
        cv2.rectangle(img_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)                                              

    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

    for contour in hulls:
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

    img_text = cv2.bitwise_and(img, img, mask=mask)
    img_text += cv2.bitwise_not(mask) # make the background appear white

    return img_hulls, img_boxes, img_text


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", default="./Input/demo", help="Path to the folder containing the input images.")
    parser.add_argument("--output_path", default="./Output/demo", help="Path to the folder which will contain the output.")
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
            
            img_hulls, img_boxes, img_text = extract_chars(img_original)

            cv2.imwrite(os.path.join(output_path, input_name), img_original)
            cv2.imwrite(os.path.join(output_path, input_name_base + "_hulls.tiff"), img_hulls)
            cv2.imwrite(os.path.join(output_path, input_name_base + "_boxes.tiff"), img_boxes)
            cv2.imwrite(os.path.join(output_path, input_name_base + "_text.tiff"), img_text)

            duration = time.time() - start
            logging.info("Processed {0} (Duration: {1:.3f} s)".format(input_name, duration)) 
            counter += 1                       

    logging.info("Processed {} images in total".format(counter))
    print("Processing done. See log file in '{}' for more details".format(output_path))

if __name__ == "__main__":
    main()