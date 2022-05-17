import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

TFRECORD_OPTION = tf.io.TFRecordOptions(compression_type="GZIP")

SYNTHETIC_DIR = "/home/cvnar1/Desktop/learningHDR_data/synthetic_data"
TRAIN_DIR = os.path.join(SYNTHETIC_DIR,"train")
VAL_DIR = os.path.join(SYNTHETIC_DIR,"validation")
TEST_DIR = os.path.join(SYNTHETIC_DIR,"test")

SAVE_DIR = "./tf_records/synthetic/test"

TFRECORD_FILE_NAME = lambda x : os.path.join(SAVE_DIR, f"{x}.tfrecords")

def convert_image_to_bytes(image):
    image_raw = image.tostring()

    return image_raw

def _bytes_feature(img):
    """Returns a bytes_list from a string / byte."""
    if isinstance(img, type(tf.constant(0))):
        img = img.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[img]))

def serialize_ds(ldr, hdr):
    feature_description = {
        "ldr":_bytes_feature(ldr),
        "hdr":_bytes_feature(hdr)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature_description))
    return example_proto.SerializeToString()

def get_images(path, save_path):
    
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    hdr_path = os.path.join(path,"*.exr")
    ldr_path = os.path.join(path,"*.jpg")
    HDRs = sorted(glob.glob(hdr_path))
    LDRs = sorted(glob.glob(ldr_path))

    for index, filepath in enumerate(HDRs):
        ref_HDR = cv2.imread(HDRs[index], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH).astype(np.float32)
        ref_LDR = cv2.imread(LDRs[index]).astype(np.float32)

        h, w, c = ref_HDR.shape
        ref_HDR = ref_HDR[:int(h/2),:,:]
        ref_LDR = ref_LDR[:int(h/2),:,:]

        ref_HDR_bytes = convert_image_to_bytes(ref_HDR)
        ref_LDR_bytes = convert_image_to_bytes(ref_LDR)

        filename = os.path.split(filepath)[-1]
        imagename = str.split(filename, sep=".")[0]

        savepath = TFRECORD_FILE_NAME(imagename)
        with tf.io.TFRecordWriter(savepath, TFRECORD_OPTION) as writer:
            print("index", index, " parsing", imagename)
            example = serialize_ds(ref_LDR_bytes, ref_HDR_bytes)
            writer.write(example)
        writer.close()


if __name__ == "__main__":
    get_images(TEST_DIR, SAVE_DIR)