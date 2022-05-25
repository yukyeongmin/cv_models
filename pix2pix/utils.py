import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

import cv2

def tone_mapping(img, a=5000.): 
    img = tf.math.multiply(a, img)
    numerator = tf.math.log(1.+ img) # 자연로그
    denominator = tf.math.log(1.+a)
    output = tf.math.divide(numerator, denominator) - 1.

    return output # -1~1

def inverse_tone_mapping(img, a=5000.):
    # network가 학습되고 나면 network의 output에 inverse tone mapping필요
    img = img + 1.
    img = tf.math.multiply(img, tf.math.log(1. + a)) #element wise
    img = tf.math.exp(img) - 1.
    img = tf.math.divide(img, a)

    return img

def wasserstein_distance(x, y, n):
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html
    
    n = n
    x = x[0] # 내림차순
    y = y[0]

    # x_sorter = np.arange(0,n)[::-1]
    # y_sorter = np.arange(0,n)[::-1]

    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    # x_sorter = tf.cast(x_sorter,dtype=tf.float32)
    # y_sorter = tf.cast(y_sorter,dtype=tf.float32)

    all_values = tf.concat([x,y],-1)
    all_values = tf.sort(all_values)

    deltas = tf.math.subtract(all_values[:,:,1:], all_values[:,:,:-1])

    x = x[:,:,::-1] # 오름차순
    y = y[:,:,::-1]
    x_cdf_indices = tf.searchsorted(x, all_values[:,:,:-1], side="right")
    y_cdf_indices = tf.searchsorted(y, all_values[:,:,:-1], side="right")
    x_cdf_indices = tf.cast(x_cdf_indices, dtype=tf.float32)
    y_cdf_indices = tf.cast(y_cdf_indices, dtype=tf.float32)

    x_cdf = tf.math.divide(x_cdf_indices, n)
    y_cdf = tf.math.divide(y_cdf_indices, n)

    output = tf.math.abs(x_cdf - y_cdf)
    output = tf.math.multiply(output, deltas)
    output = tf.math.reduce_sum(output)

    return output/128   # mean of em_distance of a batch

def compare_luminance(img1, img2):
    # 각 채널에서 상위 50개의 픽셀을 뽑아 거리 측정
    # 총 150 픽셀 비교

    n = 50
    b,w,h,c =img1.shape
    # n = w*h*c
    img1 = tf.reshape(img1,(b,-1,3))
    img2 = tf.reshape(img2,(b,-1,3))

    img1 = tf.transpose(img1,[0,2,1]) # top_k가 마지막 채널에 대해서 계산하기 때문
    img2 = tf.transpose(img2,[0,2,1])
    
    top_values1 = tf.math.top_k(img1,k=n) # return sorted_value, sorted_index
    top_values2 = tf.math.top_k(img2,k=n)
    output = wasserstein_distance(top_values1, top_values2, n)

    return output

def get_sample(ldr, hdr, target, n_sample):
    index = np.random.randint(0,hdr.shape[0],n_sample)

    selected_ldr = tf.gather(ldr, indices=index)
    selected_hdr = tf.gather(hdr, indices=index)
    selected_target = tf.gather(target, indices=index)

    return selected_ldr, selected_hdr, selected_target

def plot_images(model, test_input, target, epoch, save_dir):
    prediction = model(test_input, training=False)
    plt.figure(figsize = (15,5))

    # -1~1 -> 0~1
    display_list = [test_input[0]*0.5+0.5, target[0]*0.5+0.5, prediction[0]*0.5+0.5]
    title = ["Input Image", "Ground Truth", "Predicted Image"]

    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.title(title[i])

        plt.imshow(display_list[i])
        plt.axis("off")

    file_name = "generated_e{0:03d}".format(epoch)
    save_path = os.path.join(save_dir,file_name)
    plt.savefig(save_path)

def histogram(img, title="histogram"):
    # 원본용 histogram TODO
    hist, bin_edges = np.histogram(img, bins=10000, range=(1,1000))
    plt.plot(bin_edges[0:-1], hist)
    plt.savefig(title)
