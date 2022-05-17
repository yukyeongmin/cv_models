from tokenize import _all_string_prefixes
import numpy as np

def tone_mapping(img, a=20000): 
    # a = tf.reduce_max(img)
    img = np.log10(1 + a*img)/np.log10(1 + a) - 1
    return img

def inverse_tone_mapping(img, a=20000):
    # network가 학습되고 나면 network의 output에 inverse tone mapping필요
    img = img + 1
    img = img * np.log10(1 + a)
    img = np.power(10, img) - 1
    img = img / a

    return img