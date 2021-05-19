import numpy as np
import cv2
import math

import tensorflow as tf


def preprocess_input(x, data_format=None, version=1):
    """
    this function is taken from keras_vggface as it is this model that we need to rebuild it in tensorflow v2.4 instead of version 1
    """
    x_temp = np.copy(x)
    K = tf.keras.backend
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}
    if version == 1:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 93.5940
            x_temp[:, 1, :, :] -= 104.7624
            x_temp[:, 2, :, :] -= 129.1863
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 93.5940
            x_temp[..., 1] -= 104.7624
            x_temp[..., 2] -= 129.1863
    elif version == 2:
        if data_format == 'channels_first':
            x_temp = x_temp[:, ::-1, ...]
            x_temp[:, 0, :, :] -= 91.4953
            x_temp[:, 1, :, :] -= 103.8827
            x_temp[:, 2, :, :] -= 131.0912
        else:
            x_temp = x_temp[..., ::-1]
            x_temp[..., 0] -= 91.4953
            x_temp[..., 1] -= 103.8827
            x_temp[..., 2] -= 131.0912
    else:
        raise NotImplementedError
    return x_temp

def gamma_correction(img):

    # convert img to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #split the three values hue and saturation and value
    hue, sat, val = cv2.split(hsv)

    #Color value refers to the relative lightness or darkness of a color in image (0 to 255)
    mean = np.mean(val)
    gamma = math.log(.5*255)/math.log(mean)

    # do gamma correction on value channel to make the image more lighter
    val_gamma = np.power(val, gamma).clip(0,255).astype(np.uint8)

    if (gamma < 1):
        # combine new value channel with original hue and sat channels
        hsv_gamma = cv2.merge([hue, sat, val_gamma])

        #convert the new image to rgb again 
        new_image = cv2.cvtColor(hsv_gamma, cv2.COLOR_HSV2BGR)

        return new_image


    return img