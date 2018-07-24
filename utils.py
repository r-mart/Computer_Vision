import subprocess
import numpy as np


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