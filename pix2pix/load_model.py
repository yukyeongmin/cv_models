import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

from pix2pix_model import Generator, Discriminator
from load_tfrecords import load_tfrecords
from utils import inverse_tone_mapping, tone_mapping, histogram

TF_TRAIN_DIR = "./tf_records/synthetic/train/*.tfrecords"
TF_VAL_DIR = "./tf_records/synthetic/val/*.tfrecords"
TF_TEST_DIR = "./tf_records/synthetic/test/*.tfrecords"

NUM = 7 # 읽어올 모델
CHECKPOINT_DIR = "./pix2pix/training_checkpoints"
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, str(NUM))

BATCH_SIZE = 128
IMSHAPE = [32, 128, 3]

test_ds = load_tfrecords(TF_TEST_DIR)

generator = Generator()
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                discriminator_optimizer = discriminator_optimizer,
                                generator = generator,
                                discriminator = discriminator)

print(tf.train.latest_checkpoint(CHECKPOINT_DIR))
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir=CHECKPOINT_DIR))

file_name = "after_training"
save_path = os.path.join(CHECKPOINT_PREFIX,file_name)


# 훈련시 사용하지 않은 데이터 셋을 이용하여 모델의 성능 확인(tone_mapped ver.)
for batch, (input_image, target) in test_ds.take(10).enumerate():
    if batch.numpy==0:
        pass
    prediction_ = generator(input_image, training=False)

    # inverse tone mapping하여 exr파일로 저장
    prediction__ = inverse_tone_mapping(prediction_[0])
    save_img = prediction__.numpy()            # 32,128,3
    ori_img = target[0].numpy()
    cv2.imwrite(file_name+".exr",save_img)
    cv2.imwrite("origin.exr",ori_img)

    # histogram 확인
    histogram(ori_img,"origin")
    histogram(save_img,"prediction")

    target = target[:5]
    target_ = tone_mapping(target)
    input_image = input_image[:5] *0.5 + 0.5    # 0~1
    target_ = target_[:5] *0.5 + 0.5            # 0~1
    prediction_ = prediction_[:5] *0.5 + 0.5      # 0~1

    title =["input","target","prediction"]
    plt.figure(figsize=(15,15))
    for i in range(5):
        plt.subplot(5,3,i*3+1)
        plt.title(title[0]+str(i+1))
        plt.imshow(input_image[i])
        plt.axis("off")

        plt.subplot(5,3,i*3+2)
        plt.title(title[1]+str(i+1))
        plt.imshow(target_[i])
        plt.axis("off")

        plt.subplot(5,3,i*3+3)
        plt.title(title[2]+str(i+1))
        plt.imshow(prediction_[i])
        plt.axis("off")
    plt.savefig(save_path)


