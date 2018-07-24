import subprocess
import numpy as np
import cv2


def morph_denoise(img, size_open = 3, size_close = 2, **kwargs):
    """ Denoising using the morphological operations opening and closing """
    
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size_open, size_open))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size_close, size_close))

    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_open)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_close)    

    return img


def prepare_for_morph_filter(img):
    """ Prepares binary black on white images for using morphological filter """

    img[np.where(img == 0)] = 1
    img[np.where(img == 255)] = 0

    return img    


def restore_after_morph_filter(img):
    """ Post-processing for binary black on white images after using morphological filter """

    img[np.where(img == 0)] = 255
    img[np.where(img == 1)] = 0

    return img


def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode()