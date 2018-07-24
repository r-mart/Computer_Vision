import sys
import numpy as np

sys.path.append('..')
import utils

def test_morph_denoise():
    params = dict(
        size_open = 3,
        size_close = 5,    
        not_used_1 = 50,
        not_used_50 = "some_string"
    )

    np.random.seed(0)
    img = np.random.random((50, 50, 1))
    img = (img > 0.5).astype(float)

    img = utils.morph_denoise(img, **params)

    assert img is not None