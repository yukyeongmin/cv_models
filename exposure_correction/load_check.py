import os

import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import imageio
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.append("..")

from unet import Unet_reconst, Unet_expo
from gan import GAN
from utils.loadData import imageDataset, errorDataLoader, loadSirta


def make_gif(img_dir, save_name):
    images_path = os.listdir(img_dir)
    images_path.sort()
    frames = []
    for image_path in images_path:
        image = imageio.imread(f"{img_dir}/{image_path}")
        frames.append(image)
    imageio.mimsave(f"{img_dir}/{save_name}.gif", frames, fps=10)

def save_intermediates(layer_outs, model):

    for i in range(len(layer_outs)):
        output_batch = layer_outs[i]
        name = model.model.layers[i].name

        if len(output_batch.shape) < 4:
            print(f"{name} layer's feature.\nOutput shape of this layer is {output_batch.shape}.")
            continue

        layer_img = tf.reduce_mean(output_batch[0], axis=2)
        plt.figure(figsize=(16, 8))
        plt.imshow(layer_img)
        plt.title(f"{name} layer's feature.\nOutput shape of this layer is {output_batch.shape}.")
        plt.savefig(f'{save_dir}/{i}.png')
        plt.close()


def intermediate_model(model):
    inputs = model.generator.input
    outputs = [layer.output for layer in model.model.layers]
    intermediate_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return intermediate_model


def inspect_model(model, model_path):
    unet_inspect = intermediate_model(model)

    save_dir = f"./save/{model_path}/intermediate results"
    os.makedirs(save_dir, exist_ok=True)

    test_sample = cv2.imread(
        "/home/cvnar1/Desktop/Data/05.SIRTA/2017_single_equi_final_01_buffer/20171101_080800.jpg")
    test_sample = cv2.resize(test_sample, (IMG_SIZE[1], IMG_SIZE[0]))
    test_sample = cv2.cvtColor(test_sample, cv2.COLOR_BGR2RGB)
    test_sample = test_sample / 255.
    test_sample = tf.expand_dims(test_sample, axis=0)
    layer_outs = unet_inspect(test_sample)

    plt.figure(figsize=(16, 8))
    plt.imshow(test_sample[0])
    plt.savefig(f"{save_dir}/ori.png")
    save_intermediates(layer_outs, model)


def gaussian_pyramid_up(img, level):
    outputs = []
    outputs.append(img)
    for i in range(level):
        h, w, c = img.shape
        img = cv2.pyrUp(img, (w*2, h*2))
        outputs.append(img)
    return outputs


def reduce_pyramid(imgs):
    n = len(imgs)
    outputs = []
    for i in range(n):
        h, w, c = imgs[i].shape
        img = cv2.pyrDown(imgs[i], (w // 2, h // 2))
        outputs.append(img)
    return outputs


def get_high_freq(img, level):
    img = img.numpy()
    pyramids = gaussian_pyramid_up(img, level)
    reduces = reduce_pyramid(pyramids)

    return pyramids[level-1]-reduces[level]


def compare_reconstruction_quality(model1, model2, sample_dataset):
    return 0


def compare_error_regression_accuracy(model1, model2, sample_dataset):
    return 0


def set_plt_pos(x,y,w,h):
    fig = plt.figure(figsize=(16, 8))
    plt.get_current_fig_manager().window.setGeometry(0, 0, 1500, 500)
    return fig


if __name__ == "__main__":

    TYPE = "conv"
    NUM_DECODER = 1
    model_path = "GAN_conv/2023-05-05 12:10:08.533320"
    checkpoint = 3
    load_path = f"/home/cvnar1/Desktop/teamHDR2/exposure_correction/save/{model_path}/checkpoint/{checkpoint}"

    IMG_SIZE = (256, 1024, 3)
    BATCH_SIZE = 16

    save_dir = f'./save/{model_path}/after_training/{checkpoint}'
    os.makedirs(save_dir, exist_ok=True)

    model = GAN(IMG_SIZE, type=TYPE, num_decoder=NUM_DECODER)
    model.load_weights(load_path)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        reconstruction_loss=tf.keras.losses.MeanAbsoluteError(),
        metrics=tf.keras.losses.MeanSquaredError(),
    )

    train_generator, test_generator = loadSirta(
        csvpath="/home/cvnar1/Desktop/Data/05.SIRTA/metafiles/20230314163052_labels_norm.csv",
        high_imgdir="/home/cvnar1/Desktop/Data/05.SIRTA/2017_single_equi_final_01_buffer",
        low_imgdir="/home/cvnar1/Desktop/Data/05.SIRTA/2017_single_equi_final_03_buffer",
        batch_size=1,
        image_size=IMG_SIZE,
        shuffle=True,
        val_split=0.2
    )
    test_dataset = tf.data.Dataset.from_generator(
        generator=test_generator,
        output_signature=(tf.TensorSpec(shape=(256, 1024, 3), dtype=tf.float32),
                          tf.TensorSpec(shape=(256, 1024, 3), dtype=tf.float32))
    )
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    high_expos, low_expos = next(iter(test_dataset))

    high_encoded, high_embedding, high_intermediate_results = model.encoder(high_expos)
    high_transformed = model.residual_blocks1(high_encoded)
    re = model.decoder(high_encoded, high_intermediate_results)
    gen = model.decoder(high_transformed, high_intermediate_results)
    print(high_encoded-high_transformed)

    reconst = model.reconstruct(high_expos, type='high')
    gen = model.generate_high2low(high_expos)
    for i in range(16):
        fig_size = (16, 8)
        plt.figure(figsize=fig_size)
        plt.get_current_fig_manager().window.setGeometry(0,0,1500,500)
        plt.title('Input(high)')
        plt.imshow(high_expos[i])
        plt.show(block=False)
        plt.savefig(f'{save_dir}/{i}_input.png')

        plt.figure(figsize=fig_size)
        plt.get_current_fig_manager().window.setGeometry(100,200,1500,500)
        plt.title('Target(low)')
        plt.imshow(low_expos[i])
        plt.show(block=False)
        plt.savefig(f'{save_dir}/{i}_target.png')

        plt.figure(figsize=fig_size)
        plt.get_current_fig_manager().window.setGeometry(300, 400, 1500, 500)
        plt.imshow(reconst[i])
        plt.title('Reconstruction')
        plt.show(block=False)
        plt.savefig(f'{save_dir}/{i}_re.png')

        plt.figure(figsize=fig_size)
        plt.get_current_fig_manager().window.setGeometry(400, 600, 1500, 500)
        plt.imshow(gen[i])
        plt.title('Generation')
        plt.show(block=False)
        plt.savefig(f'{save_dir}/{i}_gen.png')

        plt.close('all')

    evaluation = model.evaluate(test_dataset)
    print(evaluation)
    exit()
